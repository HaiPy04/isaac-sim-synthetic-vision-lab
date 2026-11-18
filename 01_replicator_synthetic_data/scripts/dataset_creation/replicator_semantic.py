import omni.replicator.core as rep

# Tag the drone prim semantically so bounding box / segmentation annotators track it
drone_node = rep.get.prim_at_path("/World/shahed_drone")
with drone_node:
    rep.modify.semantics([("class", "drone")])
