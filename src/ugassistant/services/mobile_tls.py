from __future__ import annotations

from datetime import datetime, timedelta, timezone
import ipaddress
from pathlib import Path


def ensure_mobile_certificate(certificate_path: Path, key_path: Path, addresses: list[str]) -> None:
    """Create a local self-signed certificate for the mobile HTTPS listener."""
    if certificate_path.is_file() and key_path.is_file():
        return
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError as exc:
        raise RuntimeError("mobile_tls_dependency_missing") from exc
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    names: list[x509.GeneralName] = [x509.DNSName("localhost")]
    for address in addresses:
        try:
            names.append(x509.IPAddress(ipaddress.ip_address(address)))
        except ValueError:
            names.append(x509.DNSName(address))
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "UGAssistant local")])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc) - timedelta(minutes=1))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(names), critical=False)
        .sign(key, hashes.SHA256())
    )
    certificate_path.parent.mkdir(parents=True, exist_ok=True)
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
