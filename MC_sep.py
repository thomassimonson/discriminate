import typing as t
from functools import partial
from random import choices

import click
import numpy as np
from numba import jit

# TODO: Should we allow a gap character as a valid entity?

# Amino acid one-letter mapping to integer classes
AA = {c: i for i, c in enumerate("-ACDEFGHIKLMNPQRSTVWY")}
# Step is a function taking an array as a first and single argument and returning an array
Step = t.Callable[[np.ndarray], np.ndarray]
# Steps is a group of Step functions and associated probabilities
Steps = t.Sequence[t.Tuple[float, Step]]


@click.command()
@click.option('-i', '--inp', required=True, type=click.File(),
              help='a path to a file where the first line is a sequence of characters, '
                   'the second line is a true labeling, '
                   'and the third line is optional initial (guess) labeling')
@click.option('-I', '--use_init', is_flag=True, default=False,
              help='a flag whether the third line -- guess labeling -- is to be used')
@click.option('-N', '--n_steps', type=int, default=10000,
              help='a number of steps to run the algorithm')
@click.option('-c', '--classes', default=None,
              help='a sequence of comma-separated possible classes; '
                   'if not provided, it will be inferred from true labels')
@click.option('-C', '--min_members', default=None,
              help='a minimum number of members each class should have')
@click.option('-T', '--temp', type=float, default=100,
              help='unitless temperature factor')
@click.option('-m', '--prob_mut', type=float, default=0.5,
              help='probability to change a label (picked from available classes) at a random position')
@click.option('-f', '--prob_flp', type=float, default=0.5,
              help='probability to flip labels between two random positions')
@click.option('-O', '--output_frequency', type=int, default=None,
              help='output the progress for each n-th step; '
                   'if not provided, the run will have no side-effects')
@click.option('-o', '--output_buffer', type=click.File('w'), default=None,
              help='log file to write the progress if an `outuput_frequency` is provided')
def cli(inp, use_init, n_steps, classes, min_members, temp, prob_mut, prob_flp, output_frequency, output_buffer):
    """
    The tool runs MC for optimization of the alignment column separation into two subsets with minimal entropy.
    """
    # parse arguments and the input file
    column, true_labels, init_labels, classes_ = encode_input(inp, use_init)
    classes = tuple(map(int, classes.split(','))) if classes else classes_

    # construct a pool of possible steps
    if prob_flp + prob_mut != 1.0:
        raise ValueError('Sum of step probabilities must be equal to 1.0')
    steps = [
        (prob_mut, partial(mutate_label, mut_pool=classes)),
        (prob_flp, flip_pair)]

    # run the simulation
    results = run(column, init_labels, true_labels, steps, n_steps, classes, temp,
                  min_members, output_frequency, output_buffer)

    # handle the results
    col_score = column_score(results[-1], true_labels, column, (1, np.log(21)), classes)
    print(fmt_output(*results, col_score))


def run(column: np.ndarray, labels: np.ndarray, true_labels: np.ndarray, steps: Steps, n_steps: int,
        classes: t.Sequence[int], temp: float, min_members: t.Optional[int] = None,
        output_freq: t.Optional[int] = None, output_buffer: t.Optional[t.IO] = None):
    """
    Runs a simple MCMC optimizing the distribution of labels.

    :param column: Array of encoded column characters
    :param labels: Array of initial binary labels
    :param true_labels: Original labels needed to score optimized labels
    :param steps: A sequence of pairs (`prob`, `step`) where `step` is a step function
    (taking an array of labels and outputing an array of labels),
    and a `prob` is a probability to choose this step.
    It is assumed that probabilities correctly sum to 1.
    :param n_steps: A number of steps to run
    :param classes: A sequence of available classes (types of labels)
    :param min_members: A minimum number of members each class must have
    :param output_freq: Frequency of printing the output into `output_buffer`
    :param output_buffer: Passed to `file` argument of the `print` function.
    If `None`, will print to stdout.
    :param temp: The temperature factor.
    :return: best solution found during the run
    """
    # TODO: maybe require score function as an argument?
    # encapsulate arguments
    de_score_ = partial(
        de_score_numba, column=column, e_column=entropy(column), classes=classes, min_members=min_members)
    column_score_ = partial(
        column_score, labels_true=true_labels, column=column, weights=(1, np.log(21)), classes=classes)

    # unpack steps
    p_step = [s[0] for s in steps]
    steps = [s[1] for s in steps]

    # setup initial state
    current = labels.copy()
    de_current = de_score_(current)
    if de_current is None:
        raise ValueError('Initial labels yielded "None" dE score; the algorithm will not converge')
    best = 1, de_current, current

    # run the simulation
    for n in range(1, n_steps + 1):
        if output_freq and n % output_freq == 0:  # conditionally expose the current state
            print(fmt_output(n, de_current, current, column_score_(current)),
                  sep='\n', file=output_buffer)
        step = choices(steps, weights=p_step)[0]  # randomly choose a step type
        proposal = step(current)  # generate a proposal
        de_proposal = de_score_(proposal)  # calculate the score
        if de_proposal is None:  # if de_score is None skip the step
            continue
        p_accept = np.exp(-temp * (de_current - de_proposal))  # calculate acceptance probabiliy
        if np.random.rand() < p_accept:  # accept or reject?
            current, de_current = proposal, de_proposal
            if de_current > best[1]:
                best = n, de_current, current
    return best


def fmt_output(step: int, de_score_: float, labels: np.ndarray, col_score_: float) -> str:
    header = f">{step}|{round(de_score_, 4)}|{round(col_score_, 4)}"
    labels = " ".join(map(str, labels))
    return f"{header}\n{labels}"


def mutate_label(labels: np.ndarray, mut_pool: t.Sequence[int]) -> np.ndarray:
    labels = labels.copy()
    pos = np.random.randint(0, len(labels) - 1)
    mut = np.random.choice(mut_pool, 1)
    labels[pos] = mut
    return labels


def flip_pair(labels: np.ndarray) -> np.ndarray:
    labels = labels.copy()
    pos1, pos2 = np.random.randint(0, len(labels) - 1, size=2)
    labels[pos1], labels[pos2] = labels[pos2], labels[pos1]
    return labels


@jit(nopython=True, cache=True)
def entropy(classes: np.ndarray, base: int = 2) -> np.ndarray:
    counts = np.bincount(classes)
    counts = counts[counts != 0]
    norm_counts = counts / np.sum(counts)
    return np.sum(-(norm_counts * np.log(norm_counts) / np.log(base)))


@jit(nopython=True, cache=True)
def de_score_numba(labels: np.ndarray, column: np.ndarray, classes: t.Sequence[int],
                   min_members: t.Optional[int] = None, e_column: t.Optional[float] = None) -> t.Optional[float]:
    subsets = [column[labels == c] for c in classes]
    if min_members is not None:
        for s in subsets:
            if len(s) < min_members:
                return None
    weights = np.array([len(s) / len(labels) for s in subsets])
    subset_entropies = np.sum(np.array([entropy(s) for s in subsets]) * weights)
    e_col = entropy(column) if e_column is None else e_column
    return e_col - subset_entropies


@jit(nopython=True)
def column_score(labels_optimized: np.ndarray, labels_true: np.ndarray, column: np.ndarray,
                 weights: t.Tuple[float, float], classes: t.Sequence[int]) -> float:
    a, b = weights
    sep_opt = [labels_optimized[labels_true == c] for c in classes]
    sep_col = [column[labels_optimized == c] for c in classes]
    sep_opt_entropy = np.array([entropy(s) for s in sep_opt])
    sep_col_entropy = np.array([entropy(s) for s in sep_col])
    score_opt = a / np.mean(sep_opt_entropy)
    score_col = b / min(sep_col_entropy)
    return 2 * score_opt * score_col / (score_opt + score_col)


def encode_input(inp: t.Iterator[str], init_present: bool = False) \
        -> t.Tuple[np.ndarray, np.ndarray, np.ndarray, t.Tuple[int, ...]]:
    try:
        column = np.array([AA[c] for c in next(inp).rstrip('\n')])
    except KeyError:
        raise ValueError('Some one letter codes in the input column are not allowed')
    try:
        true_labels = np.array(list(map(int, next(inp).rstrip('\n').split())))
    except StopIteration:
        raise ValueError('No true labels are found')
    classes = tuple(set(true_labels))
    if init_present:
        try:
            init_labels = np.array(list(map(int, next(inp).rstrip('\n').split())))
        except StopIteration:
            raise ValueError('No initial labels are found')
    else:
        init_labels = np.random.choice(classes, size=len(true_labels))
    if not len(column) == len(true_labels) == len(init_labels):
        raise ValueError('Column and labels must be of the same length')
    return column, true_labels, init_labels, classes


if __name__ == '__main__':
    cli()
