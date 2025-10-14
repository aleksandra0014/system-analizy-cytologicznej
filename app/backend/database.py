import os
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "lbc_db")

client = AsyncIOMotorClient(MONGO_URI)
db     = client[MONGO_DB]

slides_bucket = AsyncIOMotorGridFSBucket(db, bucket_name="slides")
crops_bucket  = AsyncIOMotorGridFSBucket(db, bucket_name="crops")
xai_bucket    = AsyncIOMotorGridFSBucket(db, bucket_name="xai")

COLL = {
    "lekarze":  "lekarze",
    "pacjenci": "pacjenci",
    "slajdy":   "slajdy",
    "komorki":  "komorki",
    "gradcam":  "gradcam",
    "lime":     "lime",
}

VALIDATORS = {
    COLL["lekarze"]: {
        "validator": {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["imie", "nazwisko", "email", "created_at"],
                "properties": {
                    "lekarz_uid": {"bsonType": "string"},
                    "imie": {"bsonType": "string"},
                    "nazwisko": {"bsonType": "string"},
                    "email": {"bsonType": "string"},
                    "telefon": {"bsonType": ["string","null"]},
                    "rola": {"enum": ["doctor", "admin", "tech", "viewer"]},
                    "aktywny": {"bsonType": ["bool","null"]},
                    "password_hash": {"bsonType": ["string","null"]},
                    "created_at": {"bsonType": "date"}
                }
            }
        }
    },
    COLL["pacjenci"]: {
        "validator": {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["pacjent_uid", "created_at"],
                "properties": {
                    "pacjent_uid": {"bsonType": "string"},
                    "choroba": {"bsonType": ["string","null"]},
                    "slajd": {"bsonType": ["string","null"]},
                    "created_at": {"bsonType": "date"}
                }
            }
        }
    },
    COLL["slajdy"]: {
        "validator": {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["slajd_uid","pacjent_uid","created_at"],
                "properties": {
                    "slajd_uid": {"bsonType": "string"},
                    "pacjent_uid": {"bsonType": "string"},
                    "lekarz_uid": {"bsonType": ["string","null"]},
                    "created_at": {"bsonType": "date"},
                    "status": {"enum": ["new","processed","failed"]},
                    "overall_class": {"bsonType": ["string","null"]},
                    "slide_summary_text": {"bsonType": ["string","null"]},
                    "bbox_gridfs_name": {"bsonType": ["string","null"]},
                    "bbox_url": {"bsonType": ["string","null"]},
                    "add_info": {"bsonType": ["string","null"]}  # ⬅️ NOWE
                }
            }
        }
    },
    COLL["komorki"]: {
        "validator": {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["komorka_uid","slajd_uid","pacjent_uid","cell_id","created_at"],
                "properties": {
                    "komorka_uid": {"bsonType": "string"},
                    "slajd_uid": {"bsonType": "string"},
                    "pacjent_uid": {"bsonType": "string"},
                    "cell_id": {"bsonType": "string"},
                    "klasa": {"bsonType": ["string","null"]},
                    "probs": {"bsonType": "object"},
                    "features": {"bsonType": "object"},
                    "crop_gridfs_name": {"bsonType": ["string","null"]},
                    "crop_url": {"bsonType": ["string","null"]},
                    "created_at": {"bsonType": "date"},
                    "explanations": {"bsonType": ["object","null"]}
                }
            }
        }
    },
    COLL["gradcam"]: {
        "validator": {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["created_at"],
                "properties": {
                    "komorka_uid": {"bsonType": ["string","null"]},
                    "created_at": {"bsonType": "date"},
                    "predicted_class": {"bsonType": ["string","null"]},
                    "overlay_gridfs_name": {"bsonType": ["string","null"]},
                    "heatmap_gridfs_name": {"bsonType": ["string","null"]},
                    "activation_gridfs_name": {"bsonType": ["string","null"]},
                    "overlay_url": {"bsonType": ["string","null"]},
                    "heatmap_url": {"bsonType": ["string","null"]},
                    "activation_url": {"bsonType": ["string","null"]}
                }
            }
        }
    },
    COLL["lime"]: {
        "validator": {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["created_at","html_gridfs_name"],
                "properties": {
                    "komorka_uid": {"bsonType": ["string","null"]},
                    "created_at": {"bsonType": "date"},
                    "html_gridfs_name": {"bsonType": "string"},
                    "html_url": {"bsonType": ["string","null"]}
                }
            }
        }
    }
}

INDEXES = {
    COLL["lekarze"]: [
        ( [("email", 1)], {"unique": True} ),
        ( [("lekarz_uid", 1)], {"unique": False} ),
    ],
    COLL["pacjenci"]: [
        ( [("pacjent_uid", 1)], {"unique": True} ),
        ( [("created_at", -1)], {} ),
    ],
    COLL["slajdy"]: [
        ( [("slajd_uid", 1)], {"unique": True} ),
        ( [("pacjent_uid", 1), ("created_at", -1)], {} ),
        ( [("status", 1)], {} ),
    ],
    COLL["komorki"]: [
        ( [("komorka_uid", 1)], {"unique": True} ),
        ( [("slajd_uid", 1), ("cell_id", 1)], {"unique": True} ),
        ( [("pacjent_uid", 1), ("created_at", -1)], {} ),
        ( [("klasa", 1)], {} ),
    ],
    COLL["gradcam"]: [
        ( [("komorka_uid", 1), ("created_at", -1)], {} ),
    ],
    COLL["lime"]: [
        ( [("komorka_uid", 1), ("created_at", -1)], {} ),
    ],
}

async def ensure_collections():
    """Tworzy/aktualizuje kolekcje z walidatorami i indeksami."""
    from pymongo import MongoClient as _SyncClient
    sync = _SyncClient(MONGO_URI)[MONGO_DB]

    for cname, cfg in VALIDATORS.items():
        if cname not in sync.list_collection_names():
            sync.create_collection(cname, **cfg)
        else:
            try:
                sync.command({"collMod": cname, **cfg})
            except Exception:
                pass

    for cname, idx_list in INDEXES.items():
        coll = sync[cname]
        for keys, opts in idx_list:
            coll.create_index(keys, **opts)
