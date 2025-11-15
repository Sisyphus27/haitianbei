# Neo4j 容器创建说明

## 创建新的Neo4j容器

### 1. 停止并删除现有容器（如果存在）

```bash
# 停止现有容器
docker stop neo4j 2>$null

# 删除现有容器
docker rm neo4j 2>$null
```

### 2. 创建新的Neo4j容器

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/test123456 \
  -e NEO4J_PLUGINS='["apoc"]' \
  -v neo4j_data:/data \
  -v neo4j_logs:/logs \
  -v neo4j_import:/var/lib/neo4j/import \
  -v neo4j_plugins:/plugins \
  neo4j:latest
```

### 3. 验证容器运行状态

```bash
# 检查容器状态
docker ps | findstr neo4j

# 查看容器日志
docker logs neo4j
```

### 4. 连接信息

- **Web界面**: http://localhost:7474
- **Bolt URI**: bolt://localhost:7687
- **用户名**: neo4j
- **密码**: test123456
- **数据库名**: neo4j（默认）

### 5. 测试连接

在Python中测试连接：

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "test123456")
)

with driver.session() as session:
    result = session.run("RETURN 1 as test")
    print(result.single()["test"])  # 应该输出 1

driver.close()
```

## 注意事项

1. **数据持久化**: 使用Docker卷（neo4j_data）保存数据，即使容器删除数据也不会丢失
2. **端口映射**: 
   - 7474: HTTP端口（Web界面）
   - 7687: Bolt端口（应用程序连接）
3. **密码**: 默认密码为 `test123456`，与task2.py中的配置一致
4. **APOC插件**: 已启用APOC插件，支持更多Cypher功能

## 故障排查

如果容器无法启动：

```bash
# 查看详细日志
docker logs neo4j

# 检查端口占用
netstat -ano | findstr :7474
netstat -ano | findstr :7687

# 重新创建容器（清理所有数据）
docker stop neo4j
docker rm neo4j
docker volume rm neo4j_data neo4j_logs neo4j_import neo4j_plugins
# 然后重新执行创建命令
```

