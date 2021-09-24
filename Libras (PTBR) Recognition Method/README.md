# Libras (PTBR Hand Signal Language) Recognition 

* Proposed Solution: Use Google's Mediapipe Hands Detection to detect hands position and apply it to a image dataset for
Libras hand signals. This will generate a class - fingers coordinates dataset that can be used to train a simple NN Classifier
to recognize those signals. This is useful in order to avoid the computation of a image CNN, by only using a simple vector of coordinates
numbers, this method would be faster and requires much less processing, considering how fast Mediapipe hands detection already is.

* Advance: It's needed to make a better image dataset for the signals, with high quality and variety and also the addition of more signals.

* Advance: Use a set of frames instead a static image for hand signal classification, so it can be applied to signals that involve movement.

#### References: 

* https://github.com/google/mediapipe

* I. L. O. Bastos, M. F. Angelo and A. C. Loula, "Recognition of Static Gestures Applied to Brazilian Sign Language (Libras)," 2015 28th SIBGRAPI Conference on Graphics, Patterns and Images, Salvador, 2015, pp. 305-312.
