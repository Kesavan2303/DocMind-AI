from app import create_app

app = create_app()

if __name__ == "__main__":
    # use_reloader=False prevents double-import of LangGraph/ChromaDB on Windows
    app.run(debug=True, threaded=True, use_reloader=False)
