#!/bin/bash

echo "Running exercise 3"

dts duckiebot demo --demo_name lane_following --duckiebot_name $BOT --package_name duckietown_demos --image duckietown/dt-core:daffy-arm64v8
dts duckiebot demo --demo_name led_emitter_node --duckiebot_name $BOT --package_name led_emitter --image duckietown/dt-core:daffy-arm64v8
dts duckiebot demo --demo_name deadreckoning --duckiebot_name $BOT --package_name duckietown_demos --image duckietown/dt-core:daffy-arm64v8