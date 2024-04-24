import os
import pathlib
import platform
import re
from enum import Enum

_user_name_override = None


class PathType(Enum):
    """Determines in which format should a path be formatted.

    WINDOWS: Format with Windows style.
    LINUX: Format with Linux/Posix style.
    AUTO: Use current OS type to select either WINDOWS or LINUX.
    """
    WINDOWS = 1
    LINUX = 2
    AUTO = 3


def create_run_dir_local(run_dir_root, run_desc) -> str:
    """Create a new run dir with increasing ID number at the start."""
    run_dir_root = get_path_from_template(run_dir_root, PathType.AUTO)

    if not os.path.exists(run_dir_root):
        os.makedirs(run_dir_root)

    run_id = _get_next_run_id_local(run_dir_root)
    run_name = "{0:05d}-{1}".format(run_id, run_desc)
    run_dir = os.path.join(run_dir_root, run_name)

    if os.path.exists(run_dir):
        raise RuntimeError("The run dir already exists! ({0})".format(run_dir))

    os.makedirs(run_dir)

    return run_dir


def _get_next_run_id_local(run_dir_root: str) -> int:
    """Reads all directory names in a given directory (non-recursive) and returns the next (increasing) run id. Assumes IDs are numbers at the start of the directory names."""
    dir_names = [d for d in os.listdir(run_dir_root) if os.path.isdir(os.path.join(run_dir_root, d))]
    r = re.compile("^\\d+")  # match one or more digits at the start of the string
    run_id = 0

    for dir_name in dir_names:
        m = r.match(dir_name)

        if m is not None:
            i = int(m.group())
            run_id = max(run_id, i + 1)

    return run_id


def get_path_from_template(path_template: str, path_type: PathType = PathType.AUTO) -> str:
    """Replace tags in the given path template and return either Windows or Linux formatted path."""
    # automatically select path type depending on running OS
    if path_type == PathType.AUTO:
        if platform.system() == "Windows":
            path_type = PathType.WINDOWS
        elif platform.system() == "Linux":
            path_type = PathType.LINUX
        else:
            raise RuntimeError("Unknown platform")

    path_template = path_template.replace("<USERNAME>", get_user_name())

    # return correctly formatted path
    if path_type == PathType.WINDOWS:
        return str(pathlib.PureWindowsPath(path_template))
    elif path_type == PathType.LINUX:
        return str(pathlib.PurePosixPath(path_template))
    else:
        raise RuntimeError("Unknown platform")


def get_user_name():
    """Get the current user name."""
    if _user_name_override is not None:
        return _user_name_override
    elif platform.system() == "Windows":
        return os.getlogin()
    elif platform.system() == "Linux":
        try:
            import pwd
            return pwd.getpwuid(os.geteuid()).pw_name
        except:
            return "unknown"
    else:
        raise RuntimeError("Unknown platform")
