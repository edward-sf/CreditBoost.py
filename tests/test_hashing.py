import hashlib
from pathlib import Path

from creditboost.hashing import file_sha256

# sha256 of the literal bytes b"abc", a published test vector.
ABC_DIGEST = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_file_sha256_matches_the_known_vector_for_abc(tmp_path: Path) -> None:
    target = tmp_path / "abc.txt"
    target.write_bytes(b"abc")
    assert file_sha256(target) == ABC_DIGEST


def test_file_sha256_reads_in_chunks_so_a_file_larger_than_the_buffer_still_hashes(
    tmp_path: Path,
) -> None:
    """The implementation reads 1MB at a time; a file spanning several chunks
    must hash identically to hashing the whole payload at once."""
    payload = b"x" * (1024 * 1024 * 3 + 17)
    target = tmp_path / "big.bin"
    target.write_bytes(payload)
    assert file_sha256(target) == hashlib.sha256(payload).hexdigest()
