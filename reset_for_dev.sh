bash ./scripts/stop.sh
docker swarm leave --force
bash ./scripts/build_images.sh
bash ./scripts/start.sh
