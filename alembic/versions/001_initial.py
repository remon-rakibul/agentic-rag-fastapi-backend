"""Initial migration

Revision ID: 001_initial
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Check if tables already exist (for cases where tables were created manually)
    from sqlalchemy import inspect
    conn = op.get_bind()
    inspector = inspect(conn)
    existing_tables = inspector.get_table_names()
    
    # Create users table
    if 'users' not in existing_tables:
        op.create_table(
            'users',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('email', sa.String(), nullable=False),
            sa.Column('hashed_password', sa.String(), nullable=False),
            sa.Column('is_active', sa.Boolean(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
        op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    else:
        # Ensure indexes exist even if table exists
        existing_indexes = [idx['name'] for idx in inspector.get_indexes('users')]
        if 'ix_users_id' not in existing_indexes:
            try:
                op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
            except:
                pass
        if 'ix_users_email' not in existing_indexes:
            try:
                op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
            except:
                pass
    
    # Create documents table
    if 'documents' not in existing_tables:
        op.create_table(
            'documents',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('source_type', sa.String(), nullable=False),
            sa.Column('source_path', sa.String(), nullable=False),
            sa.Column('chunk_count', sa.Integer(), nullable=True),
            sa.Column('document_ids', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_documents_id'), 'documents', ['id'], unique=False)
        op.create_index(op.f('ix_documents_user_id'), 'documents', ['user_id'], unique=False)
    else:
        existing_indexes = [idx['name'] for idx in inspector.get_indexes('documents')]
        if 'ix_documents_id' not in existing_indexes:
            try:
                op.create_index(op.f('ix_documents_id'), 'documents', ['id'], unique=False)
            except:
                pass
        if 'ix_documents_user_id' not in existing_indexes:
            try:
                op.create_index(op.f('ix_documents_user_id'), 'documents', ['user_id'], unique=False)
            except:
                pass
    
    # Create chat_threads table
    if 'chat_threads' not in existing_tables:
        op.create_table(
            'chat_threads',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('thread_id', sa.String(), nullable=False),
            sa.Column('title', sa.String(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
            sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('thread_id')
        )
        op.create_index(op.f('ix_chat_threads_id'), 'chat_threads', ['id'], unique=False)
        op.create_index(op.f('ix_chat_threads_thread_id'), 'chat_threads', ['thread_id'], unique=True)
        op.create_index(op.f('ix_chat_threads_user_id'), 'chat_threads', ['user_id'], unique=False)
    else:
        existing_indexes = [idx['name'] for idx in inspector.get_indexes('chat_threads')]
        if 'ix_chat_threads_id' not in existing_indexes:
            try:
                op.create_index(op.f('ix_chat_threads_id'), 'chat_threads', ['id'], unique=False)
            except:
                pass
        if 'ix_chat_threads_thread_id' not in existing_indexes:
            try:
                op.create_index(op.f('ix_chat_threads_thread_id'), 'chat_threads', ['thread_id'], unique=True)
            except:
                pass
        if 'ix_chat_threads_user_id' not in existing_indexes:
            try:
                op.create_index(op.f('ix_chat_threads_user_id'), 'chat_threads', ['user_id'], unique=False)
            except:
                pass
    
    # Create chat_messages table
    if 'chat_messages' not in existing_tables:
        op.create_table(
            'chat_messages',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('thread_id', sa.String(), nullable=False),
            sa.Column('role', sa.String(), nullable=False),
            sa.Column('content', sa.Text(), nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
            sa.ForeignKeyConstraint(['thread_id'], ['chat_threads.thread_id'], ),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_chat_messages_id'), 'chat_messages', ['id'], unique=False)
        op.create_index(op.f('ix_chat_messages_thread_id'), 'chat_messages', ['thread_id'], unique=False)
    else:
        existing_indexes = [idx['name'] for idx in inspector.get_indexes('chat_messages')]
        if 'ix_chat_messages_id' not in existing_indexes:
            try:
                op.create_index(op.f('ix_chat_messages_id'), 'chat_messages', ['id'], unique=False)
            except:
                pass
        if 'ix_chat_messages_thread_id' not in existing_indexes:
            try:
                op.create_index(op.f('ix_chat_messages_thread_id'), 'chat_messages', ['thread_id'], unique=False)
            except:
                pass


def downgrade() -> None:
    op.drop_index(op.f('ix_chat_messages_thread_id'), table_name='chat_messages')
    op.drop_index(op.f('ix_chat_messages_id'), table_name='chat_messages')
    op.drop_table('chat_messages')
    
    op.drop_index(op.f('ix_chat_threads_user_id'), table_name='chat_threads')
    op.drop_index(op.f('ix_chat_threads_thread_id'), table_name='chat_threads')
    op.drop_index(op.f('ix_chat_threads_id'), table_name='chat_threads')
    op.drop_table('chat_threads')
    
    op.drop_index(op.f('ix_documents_user_id'), table_name='documents')
    op.drop_index(op.f('ix_documents_id'), table_name='documents')
    op.drop_table('documents')
    
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_table('users')

