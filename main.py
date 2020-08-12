import time
import os
import numpy as np
from src.engine import Engine
from src.assets.printables import HEADER, about_msg



def progressBar(iterable, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)

    # Initial Call
    printProgressBar(0)

    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        item.__call__()
        printProgressBar(i + 1)

    # Print New Line on Complete
    print()


engine = Engine(embeddings_file=os.path.join(os.path.dirname(__file__), './data/embeddings.data'),
                tokenizer_file=os.path.join(os.path.dirname(__file__), './data/tokenizer.data'),
                model_weights_file=os.path.join(os.path.dirname(__file__), './data/siamese.h5'))

if __name__ == '__main__':
    print(HEADER)

    items = [engine.load_embeddings, engine.load_tokenizer, engine.make_model]

    # A Nicer, Single-Call Usage
    for item in progressBar(items, prefix='Loading embeddings, tokenizers, models:', suffix='Complete', length=50):
        pass

    text = input('\nWhat would you like to learn about? (separate keywords by comma) ')

    while not text == 'exit':

        if text == 'about IRE':
            print(about_msg)
            # text = input('\nWhat would you like to learn about? ')

            text = input('\nWhat would you like to learn about? (separate keywords by comma) ')

        results = engine.make_query(keywords=text)

        summaries = list()
        keywords = list()
        tokens = list()
        for i in results:
            summaries.append(i['summary'])
            keywords.append(i['keywords'])
            tokens.append(i['tokens'][0])

        if len(summaries) == 0:
            print('No documents found. Please try another query')

            text = input('\nWhat would you like to learn about? (separate keywords by comma) ')

        else:

            q, d = engine.make_batch(query=text, document_candidates=tokens)

            print(f'\n\t{len(summaries)} documents found...\n')
            print('\tPredicting best match...\n')
            time.sleep(1)

            preds = engine.run_model(q=q, d=d)
            print(summaries[np.argmax(preds)])

            text = input('\nWhat would you like to learn about? (separate keywords by comma) ')

    print('Thanks for using the Information Retrieval Engine. Bye!')
    time.sleep(0.5)


