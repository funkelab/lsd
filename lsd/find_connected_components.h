#ifndef FIND_CONNECTED_COMPONENTS_H
#define FIND_CONNECTED_COMPONENTS_H

#include <map>

void
find_connected_components(
		const uint64_t num_nodes,
		const uint64_t* nodes_data,
		const uint64_t num_edges,
		const uint64_t* edges_data,
		const float* scores_data,
		const float threshold,
		uint64_t* components);

#endif


