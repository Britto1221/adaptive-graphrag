from src.graph.graph_retriever import ask_graph

def graph_pipeline(query:str):
    response = ask_graph(query)
    print("\nFINAL ANSWER:")
    print(response["result"])

if __name__ == "__main__":
    graph_pipeline("How is Mark Zuckerberg connected to Mukesh Ambani?")