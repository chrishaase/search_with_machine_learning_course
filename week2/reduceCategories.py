# setup countdict
countdict = {}
lines = []
# Read product-cat file
with open("../../datasets/fasttext/shuffled_normalized.txt", "r") as f:
    for line in f:
        lines.append(line)
        label = line.split()[0]
        if label in countdict:
            countdict[label] += 1
        else:
            countdict[label] = 1
# once finished write to file
with open("../../datasets/fasttext/shuffled_normalized_maxcat.txt", "w") as f:
    for line in lines:
        if countdict[line.split()[0]]>500:
            f.write(line)