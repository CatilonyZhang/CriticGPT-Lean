from pyvis.network import Network


def visualize_infotree(nodes, edges, html_filename="infotree.html"):
    """
    Create an interactive HTML visualization of the tree.
    This function uses PyVis to build a directed graph of nodes and edges,
    where synthetic nodes are displayed in red and non-synthetic in green.
    The resulting visualization is saved as an HTML file.

    Parameters
    ----------
    nodes : dict of {int : dict}
        A dictionary of node_id -> node_content.
    edges : list of tuple
        A list of tuples (parent_id, child_id, {}) representing edges in the infotree.
    html_filename : str, optional
        The filename to save the generated HTML graph (default is 'infotree.html').

    Returns
    -------
    None
        Writes the HTML file to disk and prints a confirmation message.
    """

    net = Network(height="750px", width="100%", directed=True, notebook=False)
    net.barnes_hut()

    for node_id, node_content in nodes.items():
        # Extract desired fields
        synthetic = node_content.get("stx", {}).get("range", {}).get("synthetic", False)
        goalsBefore = node_content.get("goalsBefore", "")
        goalsAfter = node_content.get("goalsAfter", "")
        pp = node_content.get("stx", {}).get("pp", "")
        start = node_content.get("stx", {}).get("range", {}).get("start", "")
        finish = node_content.get("stx", {}).get("range", {}).get("finish", "")

        # Determine border color based on synthetic
        border_color = "red" if synthetic else "green"

        # Format the display
        title = (
            f"synthetic: {synthetic}\n"
            f"start: {start}\n"
            f"finish: {finish}\n"
            f"goalsBefore: {goalsBefore}\n"
            f"pp: {pp}\n"
            f"goalsAfter: {goalsAfter}"
        )

        net.add_node(
            node_id,
            label=str(node_id),
            title=title,
            color={"background": "white", "border": border_color},
            shape="ellipse",
            size=20,
            borderWidth=3,
        )

    for source_id, target_id, _ in edges:
        net.add_edge(source_id, target_id, arrows="to", color="black")

    net.set_options(
        """
    var options = {
        "layout": {
            "hierarchical": {
                "enabled": true,
                "direction": "UD",
                "sortMethod": "directed",
                "levelSeparation": 200,
                "nodeSpacing": 150,
                "treeSpacing": 200
            }
        },
        "nodes": {
            "font": {
                "size": 14
            }
        },
        "edges": {
            "font": {
                "size": 12
            },
            "smooth": false
        },
        "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
        },
        "physics": {
            "enabled": false
        }
    }
    """
    )

    net.show(html_filename, notebook=False)
    print(f"Interactive tree saved as {html_filename}")
