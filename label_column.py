import typing as t
from itertools import islice
from random import randint

import click
import numpy as np
from Bio import SeqIO

SEQ = t.NamedTuple('sequence', [('header', str), ('seq', str)])


@click.command(name='select')
@click.option('-i', '--input_alignment', required=True,
              help='a path to the input alignment')
@click.option('-H', '--header_flag', required=True,
              help='if a sequence header contains this flag, it will be assigned class 1, and 0 otherwise')
@click.option('-R', '--random_bin_sep', default=False, is_flag=True,
              help='if flag is provided, random initial separation into binary classes is added')
@click.option('-n', '--column_number', type=int, default=None,
              help='the number of column in the alignment starting from 0; '
                   'if not provided a random column is selected')
@click.option('-o', '--output_path', type=click.File('w'), default=None,
              help='path to write an output; if not provided, stdout is used')
@click.option('-s', '--skip', type=int, default=0,
              help='if provided, this number of sequences from the start will be skipped')
def cli(input_alignment, header_flag, random_bin_sep, column_number, output_path, skip) -> None:
    """
    Command-line tool selects a columns from the `input_alignment`
    and constructs a binary labels of characters in the column
    """
    # this function just calls `select` with the given arguments and handles its output
    column, true_sep, rand_sep = label(input_alignment, header_flag, random_bin_sep, column_number, skip)
    print(column, " ".join(map(str, true_sep)), " ".join(map(str, rand_sep)), sep='\n', file=None or output_path)


def label(
        input_alignment: str,
        header_flag: t.Optional[str],
        random_bin_sep: bool = False,
        column_number: t.Optional[int] = None,
        skip: int = 0) -> t.Tuple[str, t.List[int], t.List[int]]:
    seqs = read_fasta(input_alignment, skip)
    if len(set(len(s.seq) for s in seqs)) > 1:
        raise ValueError('Sequences in the alignment are of different lengths')
    n = column_number or randint(len(seqs[0].seq) - 1)
    column = "".join((s.seq[n] for s in seqs))
    true_sep = [1 if header_flag in s.header else 0 for s in seqs]
    rand_sep = list(np.random.binomial(1, 0.5, len(column))) if random_bin_sep else []
    assert len(column) == len(true_sep)
    return column, true_sep, rand_sep


def read_fasta(path: str, skip: int = 0) -> t.List[SEQ]:
    seq_records = islice(SeqIO.parse(path, 'fasta'), skip, None)
    return [SEQ(s.id, str(s.seq)) for s in seq_records]


if __name__ == '__main__':
    cli()
