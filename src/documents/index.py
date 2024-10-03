import logging
import os
from contextlib import contextmanager
from shutil import rmtree
from typing import Optional

import tantivy
from django.conf import settings
from django.db.models import QuerySet
from filelock import FileLock
from filelock import Timeout
from guardian.shortcuts import get_users_with_perms

from documents.models import CustomFieldInstance
from documents.models import Document
from documents.models import Note
from documents.models import User

logger = logging.getLogger("paperless.index")


def get_schema():
    tokenizer = settings.INDEX_TOKENIZER

    schema_builder = tantivy.SchemaBuilder()

    schema_builder.add_integer_field("id", stored=True, indexed=True, fast=True)
    schema_builder.add_integer_field("asn", stored=True, indexed=True, fast=True)
    schema_builder.add_text_field("title", stored=True, tokenizer_name=tokenizer)
    schema_builder.add_text_field("content", stored=True, tokenizer_name=tokenizer)
    schema_builder.add_integer_field("content_length", stored=True, fast=True)
    schema_builder.add_text_field("correspondent", stored=True, tokenizer_name=tokenizer)
    schema_builder.add_text_field("tags", stored=True)
    schema_builder.add_text_field("type", stored=True)
    schema_builder.add_text_field("path", stored=True)
    schema_builder.add_date_field("created", stored=True, fast=True)
    schema_builder.add_date_field("modified", stored=True, fast=True)
    schema_builder.add_date_field("added", stored=True, fast=True)
    schema_builder.add_text_field("notes", stored=True, tokenizer_name=tokenizer)
    schema_builder.add_text_field("custom_fields", stored=True, tokenizer_name=tokenizer)
    schema_builder.add_text_field("owner", stored=True, fast=True)
    schema_builder.add_boolean_field("has_owner", stored=True, fast=True)
    schema_builder.add_text_field("original_filename", stored=True, fast=True)

    schema = schema_builder.build()

    return schema


def optimize():
    writer = index().writer()
    writer.garbage_collect_files()
    writer.wait_merging_threads()


def create_index() -> tantivy.Index:

    try:
        with FileLock(settings.INDEX_DIR / ".lock", timeout=10):
            logger.info(f"Creating new index: {settings.INDEX_DIR!s}")
            if os.path.isdir(str(settings.INDEX_DIR)):
                logger.info(f"Removing old index from path: {settings.INDEX_DIR!s}")
                rmtree(str(settings.INDEX_DIR))
            os.mkdir(str(settings.INDEX_DIR))

            return tantivy.Index(schema=get_schema(), path=str(settings.INDEX_DIR), reuse=False)
    except Timeout as e:
        logger.error(f"Timeout while trying to lock the index directory: {e!s}")
        return None


def index() -> tantivy.Index:
    try:
        return tantivy.Index(schema=get_schema(), path=str(settings.INDEX_DIR), reuse=True)
    except ValueError as e:
        if len(e.args) == 1 and e.args[0] == "Schema error: 'An index exists but the schema does not match.'":
                logger.error(
                    "Index exists but the schema does not match. "
                    "Please rebuild the index from scratch.",
                )
        else:
            raise e
    except Exception as e:
        logger.exception(f"Unable to open index: {settings.INDEX_DIR!s}")
        logger.exception(f"Caught exception: {e!s}")

    return create_index()



def last_modified():
    return os.path.getmtime(settings.INDEX_DIR)


@contextmanager
def get_writer(heap_size_mb = None):  # -> tantivy.IndexWriter:
    if not heap_size_mb:
        writer = index().writer()
    else:
        writer = index().writer(heap_size_mb * 1024 * 1024)

    try:
        yield writer
    except Exception as e:
        logger.exception(str(e))
        writer.rollback()
    else:
        writer.commit()
    finally:
        writer.wait_merging_threads()


def txn_upsert(writer, doc: Document):

    tdoc = dict(
        id=doc.pk,
        title=doc.title,
    )

    if doc.content:
        tdoc["content"] = doc.content
        tdoc["content_length"] = len(doc.content)

    if doc.created:
        tdoc["created"] = doc.created.timestamp()

    if doc.added:
        tdoc["added"] = doc.added.timestamp()

    if doc.modified:
        tdoc["modified"] = doc.modified.timestamp()

    if doc.original_filename:
        tdoc["original_filename"] = doc.original_filename

    if doc.correspondent:
        tdoc["correspondent"] = doc.correspondent.name

    if doc.document_type:
        tdoc["type"] = doc.document_type.name

    if doc.storage_path:
        tdoc["path"] = doc.storage_path.name

    if doc.owner:
        tdoc["owner"] = doc.owner.username
        tdoc["owner_id"] = doc.owner.pk
        tdoc["has_owner"] = True
    else:
        tdoc["has_owner"] = False

    users_with_perms = get_users_with_perms(
            doc,
            only_with_perms_in=["view_document"],
    )

    if users_with_perms:
        tdoc["viewer_ids"] = ",".join([str(u.id) for u in users_with_perms])


    tags = doc.tags.all()
    if len(tags) > 0:
        tdoc["tags"] = ",".join([t.name for t in tags])

    notes = Note.objects.filter(document=doc)
    if len(notes) > 0:
        tdoc["notes"] = ",".join([str(c.note) for c in notes])

    custom_fields = CustomFieldInstance.objects.filter(document=doc)
    if len(custom_fields) > 0:
        tdoc["custom_fields"] = ",".join([str(c) for c in custom_fields])

    asn = doc.archive_serial_number
    if asn is not None:
        if asn >= Document.ARCHIVE_SERIAL_NUMBER_MIN and asn <= Document.ARCHIVE_SERIAL_NUMBER_MAX:
            tdoc["asn"] = asn
        else:
            logger.error(
                f"Not indexing Archive Serial Number {asn} of document {doc.pk}. "
                f"ASN is out of range "
                f"[{Document.ARCHIVE_SERIAL_NUMBER_MIN:,}, "
                f"{Document.ARCHIVE_SERIAL_NUMBER_MAX:,}.",
            )

    writer.add_document(tantivy.Document(**tdoc))


def txn_remove(writer, doc: Document):
    txn_remove_by_id(writer, doc.pk)

def txn_remove_by_id(writer, doc_id):
    writer.delete_documents("id", doc_id)

def remove(document: Document):
    # TODO: check if autocommits
    with get_writer() as writer:
        txn_remove(writer, document)
        #writer.commit()


def upsert(document: Document):
    # TODO: check if autocommits
    with get_writer() as writer:
        txn_upsert(writer, document)
        #writer.commit()




class DelayedQuery:
    def _get_query(self):
        raise NotImplementedError  # pragma: no cover

    def _get_query_sortedby(self):
        if "ordering" not in self.query_params:
            return None, False

        field: str = self.query_params["ordering"]

        sort_fields_map = {
            "created": "created",
            "modified": "modified",
            "added": "added",
            "title": "title",
            "correspondent__name": "correspondent",
            "document_type__name": "type",
            "archive_serial_number": "asn",
            "num_notes": "num_notes",
            "owner": "owner",
        }

        if field.startswith("-"):
            field = field[1:]
            reverse = True
        else:
            reverse = False

        if field not in sort_fields_map:
            return None, False
        else:
            return sort_fields_map[field], reverse

    def __init__(
        self,
        searcher: tantivy.Searcher,
        query_params,
        page_size,
        filter_queryset: QuerySet,
    ):
        self.searcher = searcher
        self.query_params = query_params
        self.page_size = page_size
        self.saved_results = dict()
        self.first_score = None

        self.filter_queryset = filter_queryset
        self.filtered_doc_ids = filter_queryset.order_by("id").values_list("id", flat=True)

        self.content_highlighter = None
        self.notes_highlighter = None
        self.number_of_hits = 0

    def __len__(self):
        _ = self[0:1] # force calculation of number of hits
        return self.number_of_hits

    def __getitem__(self, item) -> dict:
        if item.start in self.saved_results:
            return self.saved_results[item.start]
        # from icecream import ic
        q, mask = self._get_query()

        sortedby, reverse = self._get_query_sortedby()

        if not self.content_highlighter:
            self.content_highlighter = tantivy.SnippetGenerator.create(
                searcher=self.searcher,
                query=q,
                schema=get_schema(),
                field_name="content",
            )
        if not self.notes_highlighter:
            self.notes_highlighter = tantivy.SnippetGenerator.create(
                searcher=self.searcher,
                query=q,
                schema=get_schema(),
                field_name="notes",
            )

        page = []

        search_results = self.searcher.search(
            query=q,
            offset=item.start,
            limit=self.page_size,
            order_by_field=sortedby,
            order=tantivy.Order.Desc if reverse else tantivy.Order.Asc,
        )

        if len(search_results.hits) == 0:
            self.saved_results[item.start] = page
            return page

        # from icecream import ic

        if (
            self.first_score is None
            and item.start == 0
            and len(search_results.hits) > 0
        ):
            self.first_score = search_results.hits[0][0]

        self.number_of_hits = search_results.count

        for rank_in_page, doc_score in enumerate(search_results.hits):
            score, doc_id = doc_score
            doc = self.searcher.doc(doc_id)
            d = doc.to_dict()

            if "content" in d:
                del d["content"]

            if "notes" in d:
                del d["notes"]

            d = {k: v[0] for k, v in d.items()} # tantivy returns each hit as a 1-element list of values

            d["score"] = score
            d["score_norm"] = float(score) / self.first_score
            d["rank"] = item.start + rank_in_page
            d["highlights"] = [self.content_highlighter.snippet_from_doc(doc).to_html()]
            d["note_highlights"] = [
                self.notes_highlighter.snippet_from_doc(doc).to_html(),
            ]
            page.append(d)

        self.retreived_hits = len(page)

        self.saved_results[item.start] = page
        # ic(page)
        return page


class DelayedFullTextQuery(DelayedQuery):
    def _get_query(self):
        q_str = self.query_params["query"]
        q_fields = [
            "content",
            "title",
            "correspondent",
            "tags",
            "type",
            "notes",
            "custom_fields",
        ]

        q, error = index().parse_query_lenient(
            query=q_str,
            default_field_names=q_fields,
        )
        q = tantivy.Query.boolean_query([
            (tantivy.Occur.Must, q),
            (tantivy.Occur.Must, tantivy.Query.term_set_query(get_schema(), "id", self.filtered_doc_ids)),
        ])

        return q, False


class DelayedMoreLikeThisQuery(DelayedQuery):
    def _get_query(self):
        # Requires enable scoring. How?
        more_like_doc_id = int(self.query_params["more_like_id"])
        id_lookup_query = tantivy.Query.term_query(
            schema=get_schema(),
            field_name="id",
            field_value=more_like_doc_id,
        )
        results = index().searcher().search(id_lookup_query)
        docaddr = results.hits[0][1]
        query = tantivy.Query.more_like_this_query(docaddr)


        mask = {more_like_doc_id}

        return query, mask


# class LocalDateParser(English):
#     def reverse_timezone_offset(self, d):
#         return (d.replace(tzinfo=django_timezone.get_current_timezone())).astimezone(
#             timezone.utc,
#         )

#     def date_from(self, *args, **kwargs):
#         d = super().date_from(*args, **kwargs)
#         if isinstance(d, timespan):
#             d.start = self.reverse_timezone_offset(d.start)
#             d.end = self.reverse_timezone_offset(d.end)
#         elif isinstance(d, datetime):
#             d = self.reverse_timezone_offset(d)
#         return d


# class MappedDocIdSet(DocIdSet):
#     """
#     A DocIdSet backed by a set of `Document` IDs.
#     Supports efficiently looking up if a whoosh docnum is in the provided `filter_queryset`.
#     """

#     def __init__(self, filter_queryset: QuerySet, ixreader: IndexReader) -> None:
#         super().__init__()
#         document_ids = filter_queryset.order_by("id").values_list("id", flat=True)
#         max_id = document_ids.last() or 0
#         self.document_ids = BitSet(document_ids, size=max_id)
#         self.ixreader = ixreader

#     def __contains__(self, docnum):
#         document_id = self.ixreader.stored_fields(docnum)["id"]
#         return document_id in self.document_ids

#     def __bool__(self):
#         # searcher.search ignores a filter if it's "falsy".
#         # We use this hack so this DocIdSet, when used as a filter, is never ignored.
#         return True


def autocomplete(
    ix: tantivy.Index,
    term: str,
    limit: int = 10,
    user: Optional[User] = None,
):
    return []


#     """
#     Mimics whoosh.reading.IndexReader.most_distinctive_terms with permissions
#     and without scoring
#     """
#     terms = []

#     with ix.searcher(weighting=TF_IDF()) as s:
#         qp = QueryParser("content", schema=ix.schema)
#         # Don't let searches with a query that happen to match a field override the
#         # content field query instead and return bogus, not text data
#         qp.remove_plugin_class(FieldsPlugin)
#         q = qp.parse(f"{term.lower()}*")
#         user_criterias = get_permissions_criterias(user)

#         results = s.search(
#             q,
#             terms=True,
#             filter=query.Or(user_criterias) if user_criterias is not None else None,
#         )

#         termCounts = Counter()
#         if results.has_matched_terms():
#             for hit in results:
#                 for _, match in hit.matched_terms():
#                     termCounts[match] += 1
#             terms = [t for t, _ in termCounts.most_common(limit)]

#         term_encoded = term.encode("UTF-8")
#         if term_encoded in terms:
#             terms.insert(0, terms.pop(terms.index(term_encoded)))

#     return terms


def get_permissions_criterias(user: Optional[User] = None):
    user_criterias = [tantivy.Query.Term("has_owner", False)]
    if user is not None:
        if user.is_superuser:  # superusers see all docs
            user_criterias = []
        else:
            user_criterias.append(tantivy.Query.Term("owner_id", user.id))
            user_criterias.append(
                tantivy.Query.Term("viewer_id", str(user.id)),
            )
    return user_criterias


def get_documents_stats(user: User):
    num_docs = index().searcher().num_docs

     # aggregation not yet released in tantivy
    #  total_content_length = index().searcher().aggregate(
    #     search_query=tantivy.Query.all_query(),
    #     agg_query = [{"sum": "content_length"}],
    #  )

    total_content_length = 0
    sch = index().searcher()
    results = sch.search(tantivy.Query.all_query(), limit=num_docs)
    for score, doc_id in results.hits:
        doc = sch.doc(doc_id)
        total_content_length += doc["content_length"][0]
    return num_docs, total_content_length
