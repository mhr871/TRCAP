import logging
import os
import re
import subprocess
import tempfile
import yaml
from torch.utils.tensorboard import SummaryWriter

# JVM unified-logging warning lines (e.g.
# "[0.016s][warning][os,container] Cgroup memory controller path at
# '/sys/fs/cgroup' seems to have moved to '/../../jupyter-children', detected
# limits won't be accurate") that some modern JDKs print to STDOUT (not
# stderr) when the cgroup layout doesn't match what an old JVM expects --
# observed in Colab's container. pycocoevalcap's PTBTokenizer.tokenize()
# (stanford-corenlp-3.4.1.jar, 2015) captures raw stdout and does a purely
# positional `zip(image_id, lines)` with no validation, so every such extra
# line silently shifts EVERY caption after it by one position -- an entire
# eval set can be scored against the wrong image's reference/generated text,
# consistently, with no error or warning. Confirmed empirically: see
# tools/check_tokenization.py output where pair N's "tokenized" text was
# actually pair (N-2)'s.
_JVM_LOG_LINE_RE = re.compile(r"^\[[\d.]+s?\]\[(warning|error|info)\]")


class SafePTBTokenizer:
    """Drop-in replacement for pycocoevalcap.tokenizer.ptbtokenizer.PTBTokenizer
    that filters JVM log lines out of the tokenizer's stdout before doing the
    image_id<->line pairing, so a leaked JVM warning can no longer silently
    shift every caption after it to the wrong image. Everything else
    (command, PUNCTUATIONS stripping, temp file handling) is copied from the
    original so tokenization behavior is otherwise identical.
    """

    from pycocoevalcap.tokenizer.ptbtokenizer import (
        STANFORD_CORENLP_3_4_1_JAR, PUNCTUATIONS,
    )

    def tokenize(self, captions_for_image):
        cmd = ['java', '-cp', self.STANFORD_CORENLP_3_4_1_JAR,
               'edu.stanford.nlp.process.PTBTokenizer',
               '-preserveLines', '-lowerCase']

        final_tokenized_captions_for_image = {}
        image_id = [k for k, v in captions_for_image.items() for _ in range(len(v))]
        sentences = '\n'.join([c['caption'].replace('\n', ' ')
                                for k, v in captions_for_image.items() for c in v])

        import pycocoevalcap.tokenizer.ptbtokenizer as _ptbmod
        path_to_jar_dirname = os.path.dirname(os.path.abspath(_ptbmod.__file__))
        tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=path_to_jar_dirname)
        tmp_file.write(sentences.encode())
        tmp_file.close()

        cmd.append(os.path.basename(tmp_file.name))
        p_tokenizer = subprocess.Popen(cmd, cwd=path_to_jar_dirname, stdout=subprocess.PIPE)
        token_lines = p_tokenizer.communicate(input=sentences.rstrip())[0]
        token_lines = token_lines.decode()
        raw_lines = token_lines.split('\n')
        os.remove(tmp_file.name)

        dropped = [ln for ln in raw_lines if _JVM_LOG_LINE_RE.match(ln.strip())]
        lines = [ln for ln in raw_lines if not _JVM_LOG_LINE_RE.match(ln.strip())]
        if dropped:
            print(f"[SafePTBTokenizer] dropped {len(dropped)} leaked JVM log line(s) "
                  f"from tokenizer stdout (would otherwise have shifted every "
                  f"caption after them to the wrong image): {dropped!r}")
        if len(lines) != len(image_id):
            print(f"[SafePTBTokenizer] WARNING: expected {len(image_id)} tokenized "
                  f"lines, got {len(lines)} after filtering -- pairing may still be "
                  f"wrong, investigate raw output: {raw_lines!r}")

        for k, line in zip(image_id, lines):
            if k not in final_tokenized_captions_for_image:
                final_tokenized_captions_for_image[k] = []
            tokenized_caption = ' '.join([w for w in line.rstrip().split(' ')
                                          if w not in self.PUNCTUATIONS])
            final_tokenized_captions_for_image[k].append(tokenized_caption)

        return final_tokenized_captions_for_image


def over_write_args(args, yml):
    assert os.path.isfile(yml)
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_logger(name, save_path=None, level='INFO', filename='log.txt'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, filename), encoding='utf-8')
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


class TBLog:
    """
    Construc tensorboard writer (self.writer).
    The tensorboard is saved at os.path.join(tb_dir, file_name).
    """

    def __init__(self, tb_dir, file_name, use_tensorboard=False):
        self.tb_dir = tb_dir
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            self.writer = SummaryWriter(os.path.join(self.tb_dir, file_name))
        else:
            # self.writer = CustomWriter(os.path.join(self.tb_dir, file_name))
            self.writer = None  # TODO: not implemented

    def update(self, tb_dict, it, suffix=None, mode="train"):
        """
        Args
            tb_dict: contains scalar values for updating tensorboard
            it: contains information of iteration (int).
            suffix: If not None, the update key has the suffix.
        """
        if suffix is None:
            suffix = ''
        if self.use_tensorboard:
            for key, value in tb_dict.items():
                self.writer.add_scalar(suffix + key, value, it)
        else:
            self.writer.set_epoch(it, mode)
            for key, value in tb_dict.items():
                self.writer.add_scalar(suffix + key, value)
            self.writer.plot_stats()
            self.writer.dump_stats()
