from src.cascade_builder import build_interaction_graph
from src.relation_parser import InteractionRelation


def test_action_direction_is_retweeter_to_retweeted():
    graph = build_interaction_graph(
        [InteractionRelation("tweet_a", "tweet_b", "user_a", "user_b")],
        interaction_type="retweet",
        direction="action",
    )
    assert ("tweet_a", "tweet_b") in graph.edges


def test_propagation_direction_is_retweeted_to_retweeter():
    graph = build_interaction_graph(
        [InteractionRelation("tweet_a", "tweet_b", "user_a", "user_b")],
        interaction_type="retweet",
        direction="propagation",
    )
    assert ("tweet_b", "tweet_a") in graph.edges


def test_node_user_attributes_are_preserved():
    graph = build_interaction_graph(
        [InteractionRelation("tweet_a", "tweet_b", "user_a", "user_b")],
        interaction_type="retweet",
    )
    assert graph.nodes["tweet_a"]["user_id"] == "user_a"
    assert graph.nodes["tweet_b"]["user_id"] == "user_b"


def test_duplicate_edges_are_counted():
    relations = [
        InteractionRelation("tweet_a", "tweet_b", "user_a", "user_b"),
        InteractionRelation("tweet_a", "tweet_b", "user_a", "user_b"),
    ]
    graph = build_interaction_graph(relations, interaction_type="retweet")
    assert graph.number_of_edges() == 1
    assert graph.graph["n_duplicate_relations"] == 1
    assert graph.graph["n_duplicate_edges"] == 1


def test_reply_graph_propagation_is_parent_to_reply():
    graph = build_interaction_graph(
        [InteractionRelation("reply_tweet", "parent_tweet", "reply_user", "parent_user")],
        interaction_type="reply",
        direction="propagation",
    )
    assert ("parent_tweet", "reply_tweet") in graph.edges
    assert graph.graph["interaction_type"] == "reply"
