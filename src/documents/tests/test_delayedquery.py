from django.test import TestCase
from tantivy import Query

from documents.index import get_permissions_criterias
from documents.models import User


class TestDelayedQuery(TestCase):
    def setUp(self):
        super().setUp()
        # all tests run without permission criteria, so has_no_owner query will always
        # be appended.
        self.has_no_owner = Query.Or([Query.Term("has_owner", False)])

    def _get_testset__id__in(self, param, field):
        return (
            {f"{param}__id__in": "42,43"},
            Query.And(
                [
                    Query.Or(
                        [
                            Query.Term(f"{field}_id", "42"),
                            Query.Term(f"{field}_id", "43"),
                        ],
                    ),
                    self.has_no_owner,
                ],
            ),
        )

    def _get_testset__id__none(self, param, field):
        return (
            {f"{param}__id__none": "42,43"},
            Query.And(
                [
                    Query.Not(Query.Term(f"{field}_id", "42")),
                    Query.Not(Query.Term(f"{field}_id", "43")),
                    self.has_no_owner,
                ],
            ),
        )

    def test_get_permission_criteria(self):
        # tests contains tuples of user instances and the expected filter
        tests = (
            (None, [Query.Term("has_owner", False)]),
            (User(42, username="foo", is_superuser=True), []),
            (
                User(42, username="foo", is_superuser=False),
                [
                    Query.Term("has_owner", False),
                    Query.Term("owner_id", 42),
                    Query.Term("viewer_id", "42"),
                ],
            ),
        )
        for user, expected in tests:
            self.assertEqual(get_permissions_criterias(user), expected)
