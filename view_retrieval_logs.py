#!/usr/bin/env python3
"""Utility script to view and analyze retrieval logs."""
import json
import sys
from pathlib import Path
from app.utils.retrieval_logger import RetrievalLogger


def print_log_entry(entry: dict, index: int = None):
    """Pretty print a log entry."""
    if index is not None:
        print(f"\n{'='*80}")
        print(f"Entry #{index}")
        print(f"{'='*80}")
    
    print(f"Timestamp: {entry.get('timestamp', 'N/A')}")
    print(f"Query: {entry.get('query', 'N/A')}")
    print(f"Original Question: {entry.get('original_question', 'N/A')}")
    print(f"User ID: {entry.get('user_id', 'N/A')}")
    print(f"Thread ID: {entry.get('thread_id', 'N/A')}")
    print(f"Documents Retrieved: {entry.get('num_documents_retrieved', 0)}")
    
    print(f"\nRetrieved Documents:")
    for i, doc in enumerate(entry.get('documents', []), 1):
        print(f"\n  Document {i}:")
        print(f"    Content (first 200 chars): {doc.get('content', '')[:200]}...")
        print(f"    Full Length: {doc.get('content_length', 0)} chars")
        if 'doc_id' in doc:
            print(f"    Document ID: {doc.get('doc_id')}")
        if doc.get('metadata'):
            print(f"    Metadata: {doc.get('metadata')}")
    
    if entry.get('metadata'):
        print(f"\nAdditional Metadata: {entry.get('metadata')}")


def main():
    """Main function to view logs."""
    logger = RetrievalLogger()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "latest" or command == "last":
            # Show latest N entries
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            logs = logger.read_logs(limit=limit)
            print(f"\nShowing latest {len(logs)} entries:\n")
            for i, entry in enumerate(logs, 1):
                print_log_entry(entry, index=i)
        
        elif command == "query":
            # Search by query
            if len(sys.argv) < 3:
                print("Usage: python view_retrieval_logs.py query <query_string>")
                sys.exit(1)
            query = sys.argv[2]
            logs = logger.get_logs_by_query(query)
            print(f"\nFound {len(logs)} entries for query '{query}':\n")
            for i, entry in enumerate(logs, 1):
                print_log_entry(entry, index=i)
        
        elif command == "user":
            # Filter by user_id
            if len(sys.argv) < 3:
                print("Usage: python view_retrieval_logs.py user <user_id>")
                sys.exit(1)
            user_id = int(sys.argv[2])
            logs = logger.get_logs_by_user(user_id)
            print(f"\nFound {len(logs)} entries for user {user_id}:\n")
            for i, entry in enumerate(logs, 1):
                print_log_entry(entry, index=i)
        
        elif command == "stats":
            # Show statistics
            all_logs = logger.read_logs()
            if not all_logs:
                print("No logs found.")
                return
            
            print(f"\nRetrieval Log Statistics:")
            print(f"{'='*80}")
            print(f"Total Entries: {len(all_logs)}")
            
            # Unique queries
            unique_queries = set(log.get('query', '') for log in all_logs)
            print(f"Unique Queries: {len(unique_queries)}")
            
            # Users
            users = set(log.get('user_id') for log in all_logs if log.get('user_id'))
            print(f"Unique Users: {len(users)}")
            
            # Average documents per retrieval
            avg_docs = sum(log.get('num_documents_retrieved', 0) for log in all_logs) / len(all_logs)
            print(f"Average Documents per Retrieval: {avg_docs:.2f}")
            
            # Most common queries
            query_counts = {}
            for log in all_logs:
                query = log.get('query', '')
                query_counts[query] = query_counts.get(query, 0) + 1
            
            print(f"\nTop 10 Most Common Queries:")
            for query, count in sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  '{query}': {count} times")
        
        elif command == "help":
            print("""
Retrieval Log Viewer

Usage:
  python view_retrieval_logs.py [command] [args]

Commands:
  latest [N]          Show latest N entries (default: 5)
  query <query>      Show entries matching a specific query
  user <user_id>     Show entries for a specific user
  stats              Show statistics about all logs
  help               Show this help message

Examples:
  python view_retrieval_logs.py latest 10
  python view_retrieval_logs.py query "owners of recom"
  python view_retrieval_logs.py user 16
  python view_retrieval_logs.py stats
            """)
        else:
            print(f"Unknown command: {command}")
            print("Use 'help' to see available commands")
            sys.exit(1)
    else:
        # Default: show latest 5 entries
        logs = logger.read_logs(limit=5)
        print(f"\nShowing latest {len(logs)} entries:\n")
        for i, entry in enumerate(logs, 1):
            print_log_entry(entry, index=i)


if __name__ == "__main__":
    main()

