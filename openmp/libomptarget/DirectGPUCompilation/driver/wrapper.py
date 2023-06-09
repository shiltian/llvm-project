import argparse
import os
import pathlib
import subprocess
import sys
import tempfile


cwd = os.path.dirname(os.path.realpath(__file__))

source_file_suffix = [".c", ".cpp", ".cu", ".hip", ".cc", ".cxx"]


def print_version():
    cp = subprocess.run(["clang", "--version"])
    if cp.returncode != 0:
        sys.exit(cp.returncode)


def compile_loader(loader_name, targets, macros, verbose, dry_run):
    cmd = [
        "clang",
        "-c",
        "-fopenmp",
        "-foffload-lto",
        "-fopenmp-offload-mandatory",
        "-o",
        loader_name,
    ]
    cmd += macros
    if targets:
        for arch in targets:
            cmd.append("--offload-arch={}".format(arch))
    else:
        cmd.append("--offload-arch=native")
    cmd.append(os.path.join(cwd, "Main.c"))
    if verbose:
        print(" ".join(cmd), file=sys.stderr)
    if dry_run:
        print(" ".join(cmd), file=sys.stderr)
        return
    cp = subprocess.run(cmd)
    if cp.returncode != 0:
        sys.exit(cp.returncode)


def invoke_clang(is_cpp, args, targets, verbose, dry_run):
    cmd = [
        "clang++" if is_cpp else "clang",
        "-fopenmp",
        "-foffload-lto",
        "-fopenmp-offload-mandatory",
        "-fopenmp-globalize-to-global-space",
        "-include",
        os.path.join(cwd, "UserWrapper.h"),
        "--save-temps",
        "-rdynamic",
        "-mllvm",
        "-enable-host-rpc",
        "-mllvm",
        "-openmp-opt-disable-state-machine-rewrite",
        "-mllvm",
        "-enable-canonicalize-main-function",
        "-mllvm",
        "-canonical-main-function-name=__user_main",
    ]
    if targets:
        for arch in targets:
            cmd.append("--offload-arch={}".format(arch))
    else:
        cmd.append("--offload-arch=native")
    cmd += args
    if verbose:
        print(" ".join(cmd))
    if dry_run:
        print(" ".join(cmd))
        return
    cp = subprocess.run(cmd)
    if cp.returncode != 0:
        sys.exit(cp.returncode)


def run(is_cpp=False):
    parser = argparse.ArgumentParser(
        prog="clang-gpu", description="clang LLVM GPU compiler"
    )
    parser.add_argument(
        "-c",
        action="store_true",
        help="Only run preprocess, compile, and assemble steps",
    )
    parser.add_argument(
        "-v", action="store_true", help="Show commands to run and use verbose output"
    )
    parser.add_argument(
        "--version", action="store_true", help="Print version information"
    )
    parser.add_argument(
        "-###",
        action="store_true",
        help="Print (but do not run) the commands to run for this compilation",
    )
    parser.add_argument(
        "--offload-arch",
        type=str,
        help=(
            "Specify an offloading device architecture for CUDA, HIP, or OpenMP. (e.g."
            " sm_35). If 'native' is used the compiler will detect locally installed"
            " architectures. For HIP offloading, the device architecture can be"
            " followed by target ID features delimited by a colon (e.g."
            " gfx908:xnack+:sramecc-). May be specified more than once."
        ),
        nargs="*",
    )

    args, fwd_args = parser.parse_known_args()

    if args.version:
        print_version()
        return

    if args.c:
        fwd_args.append("-c")
    if args.v:
        fwd_args.append("-v")

    dry_run = vars(args)["###"]
    loader_name = None
    temp_files = []

    if not args.c:
        tf = tempfile.NamedTemporaryFile()
        loader_name = "{}.o".format(tf.name)
        macros = []
        for arg in fwd_args:
            if arg.startswith("-D"):
                macros.append(arg)
        compile_loader(loader_name, args.offload_arch, macros, args.v, dry_run)
        fwd_args.append(loader_name)
        temp_files.append(loader_name)

    invoke_clang(is_cpp, fwd_args, args.offload_arch, args.v, dry_run)

    for f in temp_files:
        if os.path.isfile(f):
            os.unlink(f)
