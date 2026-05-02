# Sysight Wiki 升级方案

## 目标

将当前 `wiki/` 模块从文件系统 grep 升级为 SQLite FTS5 + 引用图 + 自维护能力，
参考 llmwiki 和 repositories-wiki 的设计。

## 当前状态 vs 目标

| 能力 | 当前 | 目标 |
|------|------|------|
| 存储 | 文件系统（YAML frontmatter + markdown） | 不变 |
| 搜索 | 遍历文件 grep 匹配 | SQLite FTS5 + chunking |
| 引用关系 | 无 | cites / links_to + staleness 传播 |
| 自维护 | 每次手动 append | 写入后自动解析引用、标记 stale、触发维护 |

## SQLite Schema 新增

在 `runs.sqlite` 中增加 3 张表：

```sql
-- 文本分块（~512 tokens/chunk, ~128 token overlap）
CREATE TABLE wiki_chunks (
    id TEXT PRIMARY KEY,
    page_path TEXT NOT NULL,          -- 对应 wiki 页面路径
    chunk_index INTEGER NOT NULL,
    section_title TEXT,               -- 所属 markdown 标题, e.g. "## Trigger"
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(page_path, chunk_index)
);

-- 页面引用关系
CREATE TABLE wiki_links (
    id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,         -- 来源页面
    target_path TEXT NOT NULL,         -- 被引用页面
    link_type TEXT NOT NULL CHECK (link_type IN ('links_to', 'cites')),
    UNIQUE(source_path, target_path, link_type)
);

-- FTS5 虚拟表（自动同步）
CREATE VIRTUAL TABLE wiki_fts USING fts5(
    content,
    section_title,
    content='wiki_chunks',
    content_rowid='rowid'
);

-- 触发器：自动同步 FTS
CREATE TRIGGER wiki_chunks_ai AFTER INSERT ON wiki_chunks BEGIN
    INSERT INTO wiki_fts(rowid, content, section_title)
    VALUES (new.rowid, new.content, new.section_title);
END;

CREATE TRIGGER wiki_chunks_ad AFTER DELETE ON wiki_chunks BEGIN
    INSERT INTO wiki_fts(wiki_fts, rowid, content, section_title)
    VALUES ('delete', old.rowid, old.content, old.section_title);
END;

CREATE TRIGGER wiki_chunks_au AFTER UPDATE ON wiki_chunks BEGIN
    INSERT INTO wiki_fts(wiki_fts, rowid, content, section_title)
    VALUES ('delete', old.rowid, old.content, old.section_title);
    INSERT INTO wiki_fts(rowid, content, section_title)
    VALUES (new.rowid, new.content, new.section_title);
END;
```

## 文件改动清单

### 1. `wiki/index.py` — 重写

```python
class FTSIndex:
    def __init__(self, db_path): ...
    def search(self, query, namespace=None, limit=10) -> list[SearchResult]:
        # SELECT dc.content, dc.section_title, d.title, rank as score
        # FROM wiki_chunks dc
        # JOIN wiki_fts fts ON dc.rowid = fts.rowid
        # WHERE wiki_fts MATCH ?
        # ORDER BY rank LIMIT ?
    def chunk_page(self, page_path, content): ...
        # 按 ~512 tokens 分块，记录 section_title
    def rebuild(self): ...
```

不再用 grep 遍历文件。搜索返回 snippet + section_title + score。

### 2. `wiki/store.py` — `write_page()` 增强

```python
def write_page(self, path, content, ...):
    # 1. 写文件（保持 filesystem = source of truth）
    # 2. 重新 chunk 内容 → wiki_chunks（INSERT OR REPLACE）
    # 3. 解析引用 → wiki_links（DELETE + INSERT）
    # 4. 传播 staleness：所有 links_to 该页面的页面标记 stale
```

### 3. `wiki/references.py` — 新文件

```python
def parse_references(content: str) -> tuple[list, list]:
    """解析 markdown 内容，提取两个模式的引用"""
    # [text](path.md) → links_to
    # [^1]: source.pdf → cites
    return (links, citations)

def propagate_staleness(db, target_path):
    """标记所有引用该页面的页面为 stale"""
    # UPDATE wiki_links SET stale_since = now WHERE target_path = ?

def find_stale_pages(db) -> list[str]:
    """查找所有 stale 页面"""

def find_uncited_sources(db) -> list[str]:
    """查找未被任何 wiki 页面引用的 source 文档"""
```

### 4. `wiki/ledger.py` — `init()` 扩展

在 `RunLedger.init()` 中增加 wiki_chunks / wiki_links / wiki_fts 的建表语句。

### 5. `tools/memory/search.py` — 对接 FTS

将 `memory_search` 工具从 `raise NotImplementedError` 改为调用 `FTSIndex.search()`。

## chunking 策略

```
markdown section → chunk（~512 tokens, ~128 token overlap）

例：
  ## Trigger          → chunk 0: "D2H count aligned with step count..."
  ## Fix              → chunk 1: "Check .item() .cpu() .numpy() calls..."
  ## Example          → chunk 2: "In training loop, avoid calling..."
```

每个 chunk 记录所属页面的 section_title，搜索结果能显示"在 '## Trigger' 段落找到"。

## staleness 机制

```
page A 写入 → 解析引用 → 发现 page A links_to page B, page C
                          → page A cites source.pdf
page B 被修改 → propagate_staleness(B)
             → page A.stale_since = now()
             → memory_brief 中提示 "page A may be outdated"
```

## 不变的部分

- `store.py` 的文件 CRUD API 不变
- `brief.py` 的 build_memory_brief 接口不变
- `ledger.py` 原有的 runs/patches/findings 表不变
- `promotion.py` 和 `skills.py` 不变
- filesystem 仍然是 source of truth（SQLite 是 rebuildable index）
