For classifying product names to categories:

What precision (P@1) were you able to achieve?
(10000, 0.9695, 0.9695) on normalized and category pruned data

What fastText parameters did you use?
wordngram, lr, epochs

How did you transform the product names?
normalized text as suggested

How did you prune infrequent category labels, and how did that affect your precision?
max 500, increased precision from (9980, 0.8060120240480962, 0.8060120240480962) to (10000, 0.9695, 0.9695)

How did you prune the category tree, and how did that affect your precision?
Did not do this optional exercise
