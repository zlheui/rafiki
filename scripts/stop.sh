source ./.env.sh

LOG_FILEPATH=$LOGS_FOLDER_PATH/stop.log

# Echo title with border
title() 
{
    title="| $1 |"
    edge=$(echo "$title" | sed 's/./-/g')
    echo "$edge"
    echo "$title"
    echo "$edge"
}

title "Stopping Rafiki's DB..."
docker rm -f rafiki_db || echo "Failed to stop Rafiki's DB"

title "Stopping Rafiki's Cache..."
docker rm -f rafiki_cache || echo "Failed to stop Rafiki's Cache"

title "Stopping Rafiki's Admin..."
docker rm -f rafiki_admin || echo "Failed to stop Rafiki's Admin"

title "Stopping Rafiki's Advisor..."
docker rm -f rafiki_advisor || echo "Failed to stop Rafiki's Advisor"

title "Stopping Rafiki's Drift Detector..."
docker rm -f rafiki_drift_detector || echo "Failed to stop Rafiki's Drift Detector"

title "Stopping Rafiki's Feedback Server..."
docker rm -f rafiki_feedback || echo "Failed to stop Rafiki's Feedback Server"

echo "You'll need to destroy your machine's Docker swarm & Rafiki's logs folder at $LOGS_FOLDER_PATH manually"

