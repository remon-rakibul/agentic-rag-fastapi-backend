"""add pgvector full-text search index

Adds a generated tsvector column + GIN index over the langchain-postgres
``data`` table's ``content`` column to power the sparse arm of
HybridRetriever. Also adds a btree index on the JSONB ``user_id`` extract
to keep the user-scoped sparse query fast.

Revision ID: 8b3d2e0a4c5f
Revises: 7a2c1f3e9b4d
Create Date: 2026-05-08 20:39:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '8b3d2e0a4c5f'
down_revision = '7a2c1f3e9b4d'
branch_labels = None
depends_on = None


# Match settings.VECTOR_STORE_TABLE_NAME (default 'data'). If the deployment
# uses a different name, the values can be parameterized via environment, but
# we hard-code the default here to keep migrations deterministic.
DATA_TABLE = "data"


def upgrade() -> None:
    conn = op.get_bind()

    # Only run if the langchain-postgres data table already exists (it is
    # created lazily on first ingestion via init_vectorstore_table). Guard so
    # this migration is safe to run on a fresh DB before any ingestion.
    res = conn.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_name = :name)"
        ),
        {"name": DATA_TABLE},
    ).scalar()

    if not res:
        # Table not created yet; the application will create it on first
        # ingestion. Operators should re-run this migration after first use,
        # or call the helper SQL manually.
        return

    # Generated tsvector column over the content column. langchain-postgres
    # exposes ``content`` as a stable column name on the vector store table.
    conn.execute(
        sa.text(
            f"ALTER TABLE {DATA_TABLE} "
            "ADD COLUMN IF NOT EXISTS tsv tsvector "
            "GENERATED ALWAYS AS (to_tsvector('english', content)) STORED"
        )
    )

    # GIN index for fast full-text search.
    conn.execute(
        sa.text(
            f"CREATE INDEX IF NOT EXISTS data_tsv_gin_idx "
            f"ON {DATA_TABLE} USING GIN (tsv)"
        )
    )

    # Btree index on the JSONB user_id extract so sparse queries with the
    # ``WHERE langchain_metadata->>'user_id' = ?`` filter stay fast.
    conn.execute(
        sa.text(
            f"CREATE INDEX IF NOT EXISTS data_user_id_idx "
            f"ON {DATA_TABLE} ((langchain_metadata->>'user_id'))"
        )
    )


def downgrade() -> None:
    conn = op.get_bind()
    res = conn.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_name = :name)"
        ),
        {"name": DATA_TABLE},
    ).scalar()

    if not res:
        return

    conn.execute(sa.text(f"DROP INDEX IF EXISTS data_user_id_idx"))
    conn.execute(sa.text(f"DROP INDEX IF EXISTS data_tsv_gin_idx"))
    conn.execute(sa.text(f"ALTER TABLE {DATA_TABLE} DROP COLUMN IF EXISTS tsv"))
