"""Download the reference parts these tests compare against, from NIST, into `models/`.

The parts are public-domain U.S. Government work published by the National Institute of
Standards and Technology. They are fetched rather than committed: the archives are large, and a
reference is only worth trusting if anyone can take it from its source.

    python benchmarks/reference-cad/fetch.py

Every file kept is checked against the SHA-256 the published results were measured on. A
mismatch is refused rather than used, because NIST revises its models and a comparison against a
different revision is not the comparison in the results. Python 3.8 or later, standard library
only.
"""
import hashlib
import io
import pathlib
import sys
import urllib.request
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "models"

# Each archive, and the members kept from it with the hash each was measured on.
ARCHIVES = [
    (
        # Design to Manufacturing and Inspection (D2MI) project models, 5.5 MB.
        "https://www.nist.gov/system/files/documents/el/msid/infotest/NIST-D2MI-Models.zip",
        {
            "NIST-D2MI-Models/827-9999-905.stp": "a0619330351df8a333376a3b69838c0fe02ee41307e924d36b77c26ec1f71254",
            "NIST-D2MI-Models/827-9999-905.pdf": "2d057a43340d0d30dba6f06bc931d54a57d2bd11885006dd265f1c3997aed6f2",
            "NIST-D2MI-Models/827-9999-906.stp": "1327cf9c4668b1d5bf15553ba4bba803abe93d1aa55d716eafedfa3daf6ac443",
            "NIST-D2MI-Models/827-9999-906.pdf": "a955c0fc0823aae7ea8545891c78732dd709c70b3c11a71938e854b27c867766",
        },
    ),
    (
        # MBE PMI validation and conformance test case definitions, 8.7 MB.
        "https://www.nist.gov/system/files/documents/noindex/2022/07/19/NIST_MBE_PMI_FTC_Definitions.zip",
        {
            "NIST_MBE_PMI_FTC_Definitions/nist_ftc_07_asme1_rd.stp": "b0eecb1ee17386ec0785a90dcdde52d6a152991db33aec0c69c7e1bbd30d42fa",
            "NIST_MBE_PMI_FTC_Definitions/nist_ftc_07_asme1_rd.pdf": "fcb214aa1aee70040b11eed88b65ebe42104f2ce1d0e94cb06668a75f8ec2456",
        },
    ),
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def have(member: str, expected: str) -> bool:
    path = OUT / pathlib.PurePosixPath(member).name
    return path.is_file() and sha256(path.read_bytes()) == expected


def main() -> int:
    OUT.mkdir(exist_ok=True)
    failed = 0
    for url, members in ARCHIVES:
        # Skip the download when every member is already here and right, so a rerun is free.
        wanted = {m: h for m, h in members.items() if not have(m, h)}
        if not wanted:
            print(f"have {len(members)} file(s) from {url.rsplit('/', 1)[-1]}")
            continue
        print(f"fetching {url}")
        # NIST's server answers 403 to Python's default user agent, so this one names itself.
        request = urllib.request.Request(url, headers={"User-Agent": "fieldform-reference-cad-fetch/1"})
        with urllib.request.urlopen(request, timeout=300) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        for member, expected in wanted.items():
            name = pathlib.PurePosixPath(member).name
            try:
                data = archive.read(member)
            except KeyError:
                print(f"  MISSING {member}: the archive has changed")
                failed += 1
                continue
            if sha256(data) != expected:
                print(f"  REFUSED {name}: SHA-256 {sha256(data)}, the results were measured on {expected}")
                failed += 1
                continue
            (OUT / name).write_bytes(data)
            print(f"  {name}  {len(data):,} bytes")
    if failed:
        print(f"{failed} file(s) not fetched; the published results do not apply to what NIST serves now")
        return 1
    print(f"all reference files are in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
