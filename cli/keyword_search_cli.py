import argparse
from lib.keyword_search import bm25_idf_command, build_command, search_command, tf_command, idf_command, tf_idf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser(
        "build", help="Build movie index and docmap")

    tf_parser = subparsers.add_parser(
        "tf", help="Get requested token occurences")
    tf_parser.add_argument(
        "doc_id", type=int, help="Document id for term lookup")
    tf_parser.add_argument(
        "term", type=str, help="Term to get number of occurences")

    idf_parser = subparsers.add_parser(
        "idf", help="Get inverted document frequency score for term"
    )
    idf_parser.add_argument("term", help="Term to base idf score on")

    tf_idf_parser = subparsers.add_parser(
        "tfidf", help="Get tf-idf score"
    )
    tf_idf_parser.add_argument(
        "doc_id", type=int, help="Document id for term lookup")
    tf_idf_parser.add_argument(
        "term", type=str, help="Term to get tf-idf score")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index build successfully.")
        case "tf":
            print(
                f"Number of occurences for {args.term} in doc {args.doc_id}:")
            tf_command(args.doc_id, args.term)
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tf_idf = tf_idf_command(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            bm_25_idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm_25_idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
