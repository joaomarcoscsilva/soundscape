from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def parallel_map(fn, args):
    """
    Map a function to a list of arguments in parallel.
    """
    
    with Pool(cpu_count()) as p:
        return list(tqdm(p.imap(fn, args), total=len(args) if args.__len__ else None))
