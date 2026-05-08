"""add image_assets and table_elements

Revision ID: 7a2c1f3e9b4d
Revises: 6847f52fc9be
Create Date: 2026-05-08 20:38:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '7a2c1f3e9b4d'
down_revision = '6847f52fc9be'
branch_labels = None
depends_on = None


def upgrade() -> None:
    from sqlalchemy import inspect
    conn = op.get_bind()
    inspector = inspect(conn)
    existing_tables = inspector.get_table_names()

    if 'image_assets' not in existing_tables:
        op.create_table(
            'image_assets',
            sa.Column('id', sa.String(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('document_id', sa.Integer(), nullable=True),
            sa.Column('mime_type', sa.String(), nullable=False),
            sa.Column('image_bytes', sa.LargeBinary(), nullable=False),
            sa.Column('summary', sa.Text(), nullable=False),
            sa.Column('page_number', sa.Integer(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index(op.f('ix_image_assets_user_id'), 'image_assets', ['user_id'], unique=False)
        op.create_index(op.f('ix_image_assets_document_id'), 'image_assets', ['document_id'], unique=False)

    if 'table_elements' not in existing_tables:
        op.create_table(
            'table_elements',
            sa.Column('id', sa.String(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('document_id', sa.Integer(), nullable=True),
            sa.Column('raw_markdown', sa.Text(), nullable=False),
            sa.Column('summary', sa.Text(), nullable=False),
            sa.Column('page_number', sa.Integer(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index(op.f('ix_table_elements_user_id'), 'table_elements', ['user_id'], unique=False)
        op.create_index(op.f('ix_table_elements_document_id'), 'table_elements', ['document_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_table_elements_document_id'), table_name='table_elements')
    op.drop_index(op.f('ix_table_elements_user_id'), table_name='table_elements')
    op.drop_table('table_elements')

    op.drop_index(op.f('ix_image_assets_document_id'), table_name='image_assets')
    op.drop_index(op.f('ix_image_assets_user_id'), table_name='image_assets')
    op.drop_table('image_assets')
