-- TreeEngine schema (latest = v4). Older databases are upgraded by SQLiteRepository.migrate().
-- Deviation from the V0.1 baseline doc: documents.text is stored so that a Document can be
-- fully reloaded from the database (the database is the source of truth).

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    uri TEXT,
    title TEXT NOT NULL,
    description TEXT,
    text TEXT,
    text_hash TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_uri ON documents(uri);

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    parent_id TEXT,
    depth INTEGER NOT NULL,
    position INTEGER NOT NULL,
    title TEXT NOT NULL,
    summary TEXT,
    text TEXT,
    node_type TEXT NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    start_offset INTEGER,
    end_offset INTEGER,
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY(parent_id) REFERENCES nodes(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_nodes_doc_parent ON nodes(document_id, parent_id, position);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id, position);

CREATE TABLE IF NOT EXISTS blocks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    node_id TEXT,
    position INTEGER NOT NULL,
    block_type TEXT NOT NULL,
    content TEXT NOT NULL,
    page INTEGER,
    start_offset INTEGER,
    end_offset INTEGER,
    metadata_json TEXT,
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY(node_id) REFERENCES nodes(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_blocks_node ON blocks(node_id, position);
CREATE INDEX IF NOT EXISTS idx_blocks_doc ON blocks(document_id, position);

-- rowid of blocks_fts == rowid of blocks (kept in sync by SQLiteRepository).
-- title/content are stored CJK-segmented (see treeengine.core.text.segment_for_index).
-- v2: porter stemming on top of unicode61 ("shuffle" matches "shuffling"); CJK is unaffected.
CREATE VIRTUAL TABLE IF NOT EXISTS blocks_fts USING fts5(
    block_id UNINDEXED,
    document_id UNINDEXED,
    title,
    content,
    tokenize = 'porter unicode61'
);

-- v3: Graph-lite (one-hop entity expansion). Entities are corpus-wide (normalised name);
-- entity_blocks links an entity to every block it occurs in.
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    normalized TEXT NOT NULL UNIQUE,
    type TEXT
);
CREATE TABLE IF NOT EXISTS entity_blocks (
    entity_id INTEGER NOT NULL,
    block_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    frequency INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (entity_id, block_id)
);
CREATE INDEX IF NOT EXISTS idx_entity_blocks_block ON entity_blocks(block_id);
CREATE INDEX IF NOT EXISTS idx_entity_blocks_doc ON entity_blocks(document_id, entity_id);

-- v4: per-document source / index revisions and index status (TE-013 / TE-025).
-- source_revision = "sha256:" + sha256 of the canonical text; index_revision counts the
-- (re)builds of the document's blocks; status: indexing | ready | partial | failed.
-- vector_status / graph_status: ok | failed | off.
CREATE TABLE IF NOT EXISTS document_index (
    document_id TEXT PRIMARY KEY,
    source_revision TEXT,
    index_revision INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    vector_status TEXT NOT NULL DEFAULT 'off',
    graph_status TEXT NOT NULL DEFAULT 'off',
    error TEXT,
    updated_at TEXT NOT NULL
);

