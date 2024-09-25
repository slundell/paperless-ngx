import logging
import os
from contextlib import contextmanager
from shutil import rmtree
from typing import Optional

import tantivy
from django.conf import settings
from django.db.models import QuerySet
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
    schema_builder.add_text_field("title", stored=True, tokenizer_name=tokenizer)
    schema_builder.add_text_field("content", stored=True, tokenizer_name=tokenizer)
    schema_builder.add_integer_field("asn", stored=True, indexed=True)
    schema_builder.add_text_field("correspondent", stored=True)
    schema_builder.add_integer_field(
        "correspondent_id", stored=True, indexed=True, fast=True
    )
    schema_builder.add_boolean_field("has_correspondent", stored=True, indexed=True)
    schema_builder.add_text_field("tag", stored=True)
    schema_builder.add_integer_field("tag_id", stored=True, indexed=True, fast=True)
    schema_builder.add_boolean_field("has_tag", stored=True, indexed=True, fast=True)
    schema_builder.add_text_field("type", stored=True)
    schema_builder.add_integer_field("type_id", stored=True, indexed=True, fast=True)
    schema_builder.add_boolean_field("has_type", stored=True, indexed=True)
    schema_builder.add_date_field("created", stored=True, fast=True)
    schema_builder.add_date_field("modified", stored=True)
    schema_builder.add_date_field("added", stored=True)
    schema_builder.add_text_field("path", stored=True)
    schema_builder.add_integer_field("path_id", stored=True, indexed=True, fast=True)
    schema_builder.add_boolean_field("has_path", stored=True, indexed=True)
    schema_builder.add_text_field("notes", stored=True, tokenizer_name=tokenizer)
    schema_builder.add_integer_field("num_notes", stored=True, indexed=True, fast=True)
    schema_builder.add_text_field("custom_fields", stored=True)
    schema_builder.add_integer_field(
        "custom_field_count", stored=True, indexed=True, fast=True
    )
    schema_builder.add_boolean_field("has_custom_fields", stored=True, indexed=True)
    schema_builder.add_integer_field("custom_fields_id", stored=True, indexed=True)
    schema_builder.add_text_field("owner", stored=True)
    schema_builder.add_integer_field("owner_id", stored=True)
    schema_builder.add_boolean_field("has_owner", stored=True, indexed=True)
    schema_builder.add_integer_field("viewer_id", stored=True)
    schema_builder.add_text_field("checksum", stored=True)
    schema_builder.add_text_field("original_filename", stored=True)
    schema_builder.add_boolean_field("is_shared", stored=True, indexed=True)

    schema = schema_builder.build()

    return schema


def optimize():
    writer = index().writer()
    writer.garbage_collect_files()
    writer.wait_merging_threads()


def index(recreate=False) -> tantivy.Index:
    if not recreate:
        try:
            return tantivy.Index(
                schema=get_schema(), path=str(settings.INDEX_DIR), reuse=True
            )
        except Exception as e:
            logger.exception(f"Unable to open index: {settings.INDEX_DIR!s}")
            logger.exception(f"Caught exception: {e!s}")

    logger.info(f"Creating new index: {settings.INDEX_DIR!s}")
    if os.path.isdir(str(settings.INDEX_DIR)):
        rmtree(str(settings.INDEX_DIR))
    os.mkdir(str(settings.INDEX_DIR))

    return tantivy.Index(schema=get_schema(), path=str(settings.INDEX_DIR), reuse=False)


def last_modified():
    # index(recreate=False))
    return None


@contextmanager
def get_writer():  # -> tantivy.IndexWriter:
    writer = index().writer()

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
    tags = ",".join([t.name for t in doc.tags.all()])
    tags_ids = ",".join([str(t.id) for t in doc.tags.all()])
    notes = ",".join([str(c.note) for c in Note.objects.filter(document=doc)])
    custom_fields = ",".join(
        [str(c) for c in CustomFieldInstance.objects.filter(document=doc)],
    )
    custom_fields_ids = ",".join(
        [str(f.field.id) for f in CustomFieldInstance.objects.filter(document=doc)],
    )
    asn = doc.archive_serial_number
    if asn is not None and (
        asn < Document.ARCHIVE_SERIAL_NUMBER_MIN
        or asn > Document.ARCHIVE_SERIAL_NUMBER_MAX
    ):
        logger.error(
            f"Not indexing Archive Serial Number {asn} of document {doc.pk}. "
            f"ASN is out of range "
            f"[{Document.ARCHIVE_SERIAL_NUMBER_MIN:,}, "
            f"{Document.ARCHIVE_SERIAL_NUMBER_MAX:,}.",
        )
        asn = 0
    users_with_perms = get_users_with_perms(
        doc,
        only_with_perms_in=["view_document"],
    )
    viewer_ids = ",".join([str(u.id) for u in users_with_perms])
    tdoc = dict(
        id=doc.pk,
        title=doc.title,
        content=doc.content,
        correspondent=doc.correspondent.name if doc.correspondent else None,
        correspondent_id=doc.correspondent.id if doc.correspondent else None,
        has_correspondent=doc.correspondent is not None,
        tag=tags if tags else None,
        tag_id=tags_ids if tags_ids else None,
        has_tag=len(tags) > 0,
        type=doc.document_type.name if doc.document_type else None,
        type_id=doc.document_type.id if doc.document_type else None,
        has_type=doc.document_type is not None,
        created=doc.created.timestamp(),
        added=doc.added.timestamp(),
        asn=asn,
        modified=doc.modified.timestamp(),
        path=doc.storage_path.name if doc.storage_path else None,
        path_id=doc.storage_path.id if doc.storage_path else None,
        has_path=doc.storage_path is not None,
        notes=notes,
        num_notes=len(notes),
        custom_fields=custom_fields,
        custom_field_count=len(doc.custom_fields.all()),
        has_custom_fields=len(custom_fields) > 0,
        custom_fields_id=custom_fields_ids if custom_fields_ids else None,
        owner=doc.owner.username if doc.owner else None,
        owner_id=doc.owner.id if doc.owner else None,
        has_owner=doc.owner is not None,
        viewer_id=viewer_ids if viewer_ids else None,
        checksum=doc.checksum,
        original_filename=doc.original_filename,
        is_shared=len(viewer_ids) > 0,
    )
    ddoc = {k: v for k, v in tdoc.items() if v is not None}
    writer.add_document(tantivy.Document(**ddoc))


def txn_remove(writer, doc: Document):
    txn_remove_by_id(writer, doc.pk)


def txn_remove_by_id(writer, doc_id):
    writer.delete_documents("id", [doc_id])


def remove(document: Document):
    # TODO: check if autocommits
    with get_writer() as writer:
        txn_remove(writer, document)


def upsert(document: Document):
    # TODO: check if autocommits
    with get_writer() as writer:
        txn_upsert(writer, document)


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

        self.content_highlighter = None
        self.notes_highlighter = None

    def __len__(self):
        page = self[0:1]
        return len(page)

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
            del d["content"]
            del d["notes"]
            d = {k: v[0] for k, v in d.items()}

            d["score"] = score
            d["score_norm"] = float(score) / self.first_score
            d["rank"] = item.start + rank_in_page
            d["highlights"] = [self.content_highlighter.snippet_from_doc(doc).to_html()]
            d["note_highlights"] = [
                self.notes_highlighter.snippet_from_doc(doc).to_html()
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
            "tag",
            "type",
            "notes",
            "custom_fields",
        ]

        q, error = index().parse_query_lenient(
            query=q_str,
            default_field_names=q_fields,
        )
        # from icecream import ic
        # ic(q_str, q, error)

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

        # q = query.Or(
        #     [query.Term("content", word, boost=weight) for word, weight in kts],
        # )
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
