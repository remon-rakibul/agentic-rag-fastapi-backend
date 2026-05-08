"""add_token_blacklist_table

Revision ID: 6847f52fc9be
Revises: 001_initial
Create Date: 2025-11-20 13:05:25.270384

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '6847f52fc9be'
down_revision = '001_initial'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Check if table already exists
    from sqlalchemy import inspect
    conn = op.get_bind()
    inspector = inspect(conn)
    existing_tables = inspector.get_table_names()
    
    if 'token_blacklist' not in existing_tables:
        op.create_table(
            'token_blacklist',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('token', sa.String(), nullable=False),
            sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_token_blacklist_id'), 'token_blacklist', ['id'], unique=False)
        op.create_index(op.f('ix_token_blacklist_token'), 'token_blacklist', ['token'], unique=True)
        op.create_index(op.f('ix_token_blacklist_expires_at'), 'token_blacklist', ['expires_at'], unique=False)
    else:
        # Ensure indexes exist even if table exists
        existing_indexes = [idx['name'] for idx in inspector.get_indexes('token_blacklist')]
        if 'ix_token_blacklist_id' not in existing_indexes:
            try:
                op.create_index(op.f('ix_token_blacklist_id'), 'token_blacklist', ['id'], unique=False)
            except:
                pass
        if 'ix_token_blacklist_token' not in existing_indexes:
            try:
                op.create_index(op.f('ix_token_blacklist_token'), 'token_blacklist', ['token'], unique=True)
            except:
                pass
        if 'ix_token_blacklist_expires_at' not in existing_indexes:
            try:
                op.create_index(op.f('ix_token_blacklist_expires_at'), 'token_blacklist', ['expires_at'], unique=False)
            except:
                pass


def downgrade() -> None:
    op.drop_index(op.f('ix_token_blacklist_expires_at'), table_name='token_blacklist')
    op.drop_index(op.f('ix_token_blacklist_token'), table_name='token_blacklist')
    op.drop_index(op.f('ix_token_blacklist_id'), table_name='token_blacklist')
    op.drop_table('token_blacklist')

