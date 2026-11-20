# Next-course prediction on 1000 held-out students (5636 predictions)

Each test student's history is cut after every course; the method ranks all courses not yet taken and we record the rank of the course the student really took next. 'Last course is shared' is the 3834 cases where the last course leads to different next courses on different planted paths, so the earlier history matters. The oracle is not a usable method: it is told each student's planted path, so it marks how far noise in the data (electives, swaps, detours) caps any method.

| method | hits@1 | hits@3 | hits@5 | MRR | hits@1 when last course is shared | top-1 has prereqs met |
|---|---|---|---|---|---|---|
| popularity | 0.183 | 0.344 | 0.486 | 0.328 | 0.199 | 100.0% |
| last course (Markov) | 0.524 | 0.774 | 0.836 | 0.670 | 0.498 | 89.4% |
| co-occurrence (whole history) | 0.438 | 0.611 | 0.777 | 0.563 | 0.432 | 86.0% |
| *oracle: knows the student's planted path (reference)* | 0.799 | 0.837 | 0.855 | 0.831 | 0.815 | 100.0% |
| **CourseGPT (this project)** | 0.719 | 0.882 | 0.926 | 0.811 | 0.698 | 99.9% |
