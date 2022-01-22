import logging
import os


def init_logger() -> None:
    """
    Init logger
    """
    if not os.path.exists('logs'):
        os.mkdir('logs')

    logging.basicConfig(
        level=logging.DEBUG,
        filename='logs/app.log',
        format='%(asctime)s %(name)s: %(levelname)-8s %(message)s',
        datefmt='%y-%m-%d %H:%M',
        filemode='w+'
    )

    formatter = logging.Formatter('%(asctime)s %(name)s: %(levelname)-8s %(message)s')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    # disable matplotlib info logs
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
