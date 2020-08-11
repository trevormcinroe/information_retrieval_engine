import time
import os
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
                model_weights_file=None)

if __name__ == '__main__':
    print(HEADER)

    # items = [x for x in range(2)]
    items = [engine.load_embeddings, engine.load_tokenizer, engine.make_model]

    # A Nicer, Single-Call Usage
    for item in progressBar(items, prefix='Loading embeddings, tokenizers, models:', suffix='Complete', length=50):
        # Do stuff...
        # engine.load_embeddings()
        pass
        # time.sleep(2)

    text = input('\nWhat would you like to learn about? (separate keywords by comma) ')

    while not text == 'exit':

        if text == 'about IRE':
            print(about_msg)
            # text = input('\nWhat would you like to learn about? ')

        results = engine.make_query(keywords=text)

        for i in results:
            print()
            print(i['summary'])
            print()
            break

        text = input('\nWhat would you like to learn about? (separate keywords by comma) ')

    print('Thanks for using the Information Retrieval Engine. Bye!')
    time.sleep(0.5)


