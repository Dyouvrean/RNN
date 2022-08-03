def get_corpus(file):
    f = open(file, "r", encoding="utf8")
    corpus = {}
    for line in f:
        wordlist = line.split(" ")[:-1]
        corpus[wordlist[0]] = wordlist[1:]
    return corpus
