import os
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from pymongo import MongoClient as SyncMongoClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "lbc_db2")

client: AsyncIOMotorClient | None = None
db = None

slides_bucket: AsyncIOMotorGridFSBucket | None = None
crops_bucket:  AsyncIOMotorGridFSBucket | None = None
xai_bucket:    AsyncIOMotorGridFSBucket | None = None

COLL = {
    "doctors":  "doctors",
    "patients": "patients",
    "slides":   "slides",
    "cells":  "cells",
    "gradcam":  "cells", 
    "lime":     "cells",
    "access": "slides",
}

VALIDATORS = {
    COLL["doctors"]: {
        "validator": {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["doctor_uid", "name", "surname", "email", "created_at"],
                "properties": {
                    "doctor_uid": {"bsonType": "string"},
                    "name": {"bsonType": "string"},
                    "surname": {"bsonType": "string"},
                    "email": {"bsonType": "string"},
                    "phone": {"bsonType": ["string","null"]},
                    "role": {"enum": ["doctor", "admin", "tech", "viewer"]},
                    "active": {"bsonType": ["bool","null"]},
                    "password_hash": {"bsonType": ["string","null"]},
                    "created_at": {"bsonType": "date"}
                }
            }
        }
    },
    COLL["patients"]: {
        "validator": {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["patient_uid", "created_at"],
                "properties": {
                    "patient_uid": {"bsonType": "string"},
                    "slide": {"bsonType": ["string","null"]},
                    "created_at": {"bsonType": "date"}
                }
            }
        }
    },
    COLL["slides"]: {
        "validator": {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["slide_uid","patient_uid","created_at"],
                "properties": {
                    "slide_uid": {"bsonType": "string"},
                    "patient_uid": {"bsonType": "string"},
                    "doctor_uid": {"bsonType": ["string","null"]},
                    "created_at": {"bsonType": "date"},
                    "status": {"enum": ["new","processed","failed"]},
                    "overall_class": {"bsonType": ["string","null"]},
                    "slide_summary_text": {"bsonType": ["string","null"]},
                    "bbox_gridfs_name": {"bsonType": ["string","null"]},
                    "bbox_url": {"bsonType": ["string","null"]},
                    "add_info": {"bsonType": ["string","null"]},
                    "probability": {"bsonType": ["object","null"]},
                    "access": {
                        "bsonType": ["array", "null"],
                        "items": {
                            "bsonType": "object",
                            "required": ["doctor_uid", "granted_by", "granted_at", "role", "active"],
                            "properties": {
                                "doctor_uid": {"bsonType": "string"},
                                "role": {"enum": ["owner", "viewer"]}, 
                                "granted_by": {"bsonType": "string"},
                                "granted_at": {"bsonType": "date"},
                                "revoked_at": {"bsonType": ["date","null"]},
                                "active": {"bsonType": "bool"},
                                "note": {"bsonType": ["string","null"]}
                            }
                        }
                    }
                }
            }
        }
    },
    COLL["cells"]: {
        "validator": {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["cell_uid","slide_uid","patient_uid","cell_id","created_at"],
                "properties": {
                    "cell_uid": {"bsonType": "string"},
                    "slide_uid": {"bsonType": "string"},
                    "patient_uid": {"bsonType": "string"},
                    "cell_id": {"bsonType": "string"},
                    "class": {"bsonType": ["string","null"]},
                    "probs": {"bsonType": "object"},
                    "features": {"bsonType": "object"},
                    "crop_gridfs_name": {"bsonType": ["string","null"]},
                    "crop_url": {"bsonType": ["string","null"]},
                    "created_at": {"bsonType": "date"},
                    "explanations": {"bsonType": ["string","null"]},
                    "gradcam_data": {
                        "bsonType": ["object", "null"],
                        "properties": {
                            "created_at": {"bsonType": "date"},
                            "predicted_class": {"bsonType": ["string","null"]},
                            "overlay_gridfs_name": {"bsonType": ["string","null"]},
                            "heatmap_gridfs_name": {"bsonType": ["string","null"]},
                            "activation_gridfs_name": {"bsonType": ["string","null"]},
                            "overlay_url": {"bsonType": ["string","null"]},
                            "heatmap_url": {"bsonType": ["string","null"]},
                            "activation_url": {"bsonType": ["string","null"]}
                        }
                    },
                    "lime_data": {
                        "bsonType": ["object", "null"],
                        "required": ["html_gridfs_name"],
                        "properties": {
                            "created_at": {"bsonType": "date"},
                            "html_gridfs_name": {"bsonType": "string"},
                            "html_url": {"bsonType": ["string","null"]}
                        }
                    }
                }
            }
        }
    }
}

INDEXES = {
    COLL["doctors"]: [
        ( [("email", 1)], {"unique": True} ),
        ( [("doctor_uid", 1)], {"unique": False} ),
    ],
    COLL["patients"]: [
        ( [("patient_uid", 1)], {"unique": True} ),
        ( [("created_at", -1)], {} ),
    ],
    COLL["slides"]: [
        ( [("slide_uid", 1)], {"unique": True} ),
        ( [("patient_uid", 1), ("created_at", -1)], {} ),
        ( [("status", 1)], {} ),
        ( [("access.doctor_uid", 1), ("access.active", 1)], {} ),
    ],
    COLL["cells"]: [
        ( [("cell_uid", 1)], {"unique": True} ),
        ( [("slide_uid", 1), ("cell_id", 1)], {"unique": True} ),
        ( [("patient_uid", 1), ("created_at", -1)], {} ),
        ( [("class", 1)], {} ),
    ],
}

async def connect() -> None:
    global client, db, slides_bucket, crops_bucket, xai_bucket
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[MONGO_DB]
    slides_bucket = AsyncIOMotorGridFSBucket(db, bucket_name="slides")
    crops_bucket  = AsyncIOMotorGridFSBucket(db, bucket_name="crops")
    xai_bucket    = AsyncIOMotorGridFSBucket(db, bucket_name="xai")

async def disconnect() -> None:
    global client
    if client:
        client.close()

async def ensure_collections():
    sync = SyncMongoClient(MONGO_URI)[MONGO_DB]

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