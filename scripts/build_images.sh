# Echo title with border
title() 
{
    title="| $1 |"
    edge=$(echo "$title" | sed 's/./-/g')
    echo "$edge"
    echo "$title"
    echo "$edge"
}

# Build Rafiki's images

title "Building Rafiki Admin's image..."
docker build -t $RAFIKI_IMAGE_ADMIN:$RAFIKI_VERSION -f ./dockerfiles/admin.Dockerfile $PWD || exit 1 
title "Building Rafiki Advisor's image..."
docker build -t $RAFIKI_IMAGE_ADVISOR:$RAFIKI_VERSION -f ./dockerfiles/advisor.Dockerfile $PWD || exit 1 
title "Building Rafiki Worker's image..."
docker build -t $RAFIKI_IMAGE_WORKER:$RAFIKI_VERSION -f ./dockerfiles/worker.Dockerfile $PWD || exit 1 
title "Building Rafiki Predictor's image..."
docker build -t $RAFIKI_IMAGE_PREDICTOR:$RAFIKI_VERSION -f ./dockerfiles/predictor.Dockerfile $PWD || exit 1 
title "Building Rafiki Drift Detector's image..."
docker build -t $RAFIKI_IMAGE_DRIFT_DETECTOR:$RAFIKI_VERSION -f ./dockerfiles/drift_detector.Dockerfile $PWD || exit 1 
title "Building Rafiki Feedback Server's image..."
docker build -t $RAFIKI_IMAGE_FEEDBACK:$RAFIKI_VERSION -f ./dockerfiles/feedback.Dockerfile $PWD || exit 1 
title "Building Rafiki Data Repository Server's image..."
docker build -t $RAFIKI_IMAGE_DATA_REPOSITORY:$RAFIKI_VERSION -f ./dockerfiles/data_repository.Dockerfile $PWD || exit 1 
echo "Finished building all Rafiki's images successfully!"