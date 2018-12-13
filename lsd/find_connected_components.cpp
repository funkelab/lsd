#include <vector>
#include <map>
#include <iostream>
#include <cstdint>
#include <boost/pending/disjoint_sets.hpp>
#include "find_connected_components.h"

void
find_connected_components(
		const uint64_t num_nodes,
		const uint64_t* nodes_data,
		const uint64_t num_edges,
		const uint64_t* edges_data,
		const float* scores_data,
		const float threshold,
		uint64_t* components) {

	std::vector<std::size_t> rank(num_nodes);
	std::vector<std::size_t> parent(num_nodes);
	std::map<uint64_t, std::size_t> node_to_set;
	boost::disjoint_sets<std::size_t*, std::size_t*> sets(&rank[0], &parent[0]);

	// create a set for each node
	for (std::size_t i = 0; i < num_nodes; i++) {

		node_to_set[nodes_data[i]] = i;
		sets.make_set(i);
	}

	// for each edge <= threshold
	std::size_t num_edges_skipped = 0;
	for (std::size_t i = 0; i < num_edges; i++) {

		if (scores_data[i] > threshold)
			continue;

		auto it_u = node_to_set.find(edges_data[2*i]);
		auto it_v = node_to_set.find(edges_data[2*i + 1]);

		if (it_u == node_to_set.end() || it_v == node_to_set.end()) {

			num_edges_skipped++;
			continue;
		}

		// unite sets
		std::size_t u = it_u->second;
		std::size_t v = it_v->second;
		sets.union_set(u, v);
	}

	if (num_edges_skipped > 0)
		std::cout << "skipped " << num_edges_skipped << " edges that had no corresponding node" << std::endl;

	// for each node
	for (std::size_t i = 0; i < num_nodes; i++) {

		// find set representative
		components[i] = sets.find_set(i);
	}
}

