# Dataset label imbalance

The S008 and S009 datasets include capture locations where a specific look angle never produces a measurable signal. These cases typically happen when an obstruction—such as a building, wall, or other large structure—completely blocks the radio path. In the resulting label files, the “missing” angle is flagged as non-operational rather than filled with noise, so the class distribution for that angle is empty.

When training or evaluating models, this manifests as a severe imbalance across the look-angle labels: some angles have a normal number of samples while the blocked angles contribute zero. Treat these entries as special cases (e.g., by filtering them out or treating them separately) rather than trying to force a prediction for a label that was never observed during data collection.
