This is a quick and dirty implementation of AVF, entropy scoring, "shrimp" (simple Krimp encoding only 1's) and its dual "zimp" (encoding only zeros).  The entropy score is the sum of the shrimp and zimp scores.

Data formats:

- context.csv: a context file.  The first column is assumed to be the object id, the remaining ones are considered attributes.

- score.csv: a score file.  The first column is the object id and the second is the score.

- groundtruth.csv: a ground truth file.  The first column is the uuid of part of the ground truth and the second is the type of node.

- rank.csv: a ranking file.  The first column is the object id (= part of the ground truth), the second is the score, and the third is the object's rank.  Currently tied scores lead to different ranks, for example if the scores are 10, 9 and 9 then the ranks will be 1,2,3 rather than 1,3,3 or 1,2,2.

All of these have a header line at the beginning, though in some cases (such as ground truth files) it is just ignored.

Modules:

- model.py.  Defines a generic model that maintains counts of attributes, and several subclasses that perform different kinds of scoring.

Executables:

- join.py.  Takes two context files and joins them along the object id.  Uniquifies the attribute names in case of overlap.

- ad.py.  Runs an anomaly detection technique on a context file, producing a (sorted) score file.

- check.py.  Takes a score file and a ground truth file and produces a ranking file showing the scores and ranks of the detected ground truth uuids (of a given type, or of all types if none is specified).

- run.py.  Runs a collection of tests.  No attempt is made to avoid recomputing results whose source data is unchanged.  Takes an optional argument "--dir" that means that only tests in a specific directory are run.

run.py assumes data is stored in a directory structure like this:

- sourcename (e.g. cadets)
    - scenarioname (e.g. bovia)
        - context.csv and groundtruth.csv

The expected contents are shown in a nested value in run.py.  We should adjust this to match the structure of the DataShare data so that we can just copy it over and run.


test/ directory contains some simple / basic tests for scoring and ranking.
