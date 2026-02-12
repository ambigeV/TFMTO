# DDMTOLab 更新发布完整流程

## 📋 目录
1. [版本更新准备](#1-版本更新准备)
2. [代码提交到 GitHub](#2-代码提交到-github)
3. [发布到 PyPI](#3-发布到-pypi)
4. [创建 GitHub Release](#4-创建-github-release)
5. [常见问题](#5-常见问题)

---

## 1. 版本更新准备

### 1.1 决定版本号

遵循语义化版本规范 (Semantic Versioning)：`MAJOR.MINOR.PATCH`

- **MAJOR** (主版本号): 不兼容的 API 修改
- **MINOR** (次版本号): 向下兼容的功能性新增
- **PATCH** (修订号): 向下兼容的问题修正

**例子：**
- `1.0.0` → `1.0.1` (bug 修复)
- `1.0.1` → `1.1.0` (新增功能)
- `1.1.0` → `2.0.0` (重大改变)

### 1.2 更新版本号（3个文件）

**文件 1: `src/ddmtolab/__init__.py`**
```python
__version__ = "1.1.0"  # 修改这里
```

**文件 2: `pyproject.toml`**
```toml
version = "1.1.0"  # 修改这里
```

**文件 3: `conda/meta.yaml`** (如果未来发布 conda)
```yaml
{% set version = "1.1.0" %}  # 修改这里
```

### 1.3 更新 CHANGELOG（推荐）

创建或更新 `CHANGELOG.md`：

```markdown
# Changelog

## [1.1.0] - 2025-01-25

### Added
- 新增 XX 算法
- 新增 YY 功能

### Changed
- 改进 ZZ 性能
- 优化内存使用

### Fixed
- 修复 AA 问题
- 修复 BB bug

## [1.0.0] - 2025-01-24

### Added
- 初始发布
- 60+ 优化算法
- 180+ 基准问题
```

---

## 2. 代码提交到 GitHub

### 2.1 检查修改

```bash
# 查看修改的文件
git status

# 查看具体修改内容
git diff
```

### 2.2 提交代码

```bash
# 添加所有修改
git add .

# 或选择性添加
git add src/ddmtolab/
git add pyproject.toml
git add CHANGELOG.md

# 提交（写清楚改了什么）
git commit -m "Release v1.1.0: Add new algorithms and fix bugs"

# 推送到 GitHub
git push origin main
```

### 2.3 最佳实践提交信息

```bash
# 功能开发
git commit -m "feat: add new PSO algorithm"

# Bug 修复
git commit -m "fix: resolve memory leak in MTBO"

# 文档更新
git commit -m "docs: update installation guide"

# 性能改进
git commit -m "perf: optimize convergence speed"

# 版本发布
git commit -m "release: bump version to 1.1.0"
```

---

## 3. 发布到 PyPI

### 3.1 测试本地安装

```bash
# 激活环境
conda activate ddmtolab

# 进入项目目录
cd D:\DDMTOLab

# 开发模式安装
pip install -e .

# 测试导入
python -c "import ddmtolab; print(ddmtolab.__version__)"

# 测试核心功能
python -c "from ddmtolab.Methods.mtop import MTOP; print('OK')"
```

### 3.2 清理旧构建

```bash
# 清理
rmdir /s /q build
rmdir /s /q dist
for /d %i in (*.egg-info) do @rmdir /s /q "%i"
```

### 3.3 构建新版本

```bash
# 确保工具是最新的
pip install --upgrade build twine

# 构建
python -m build

# 检查
twine check dist/*
```

### 3.4 先上传到 TestPyPI（可选但推荐）

```bash
# 测试上传
twine upload --repository testpypi dist/*

# 测试安装
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ddmtolab

# 测试
python -c "import ddmtolab; print(ddmtolab.__version__)"
```

### 3.5 上传到正式 PyPI

```bash
# 上传
twine upload dist/*

# 会显示
# View at: https://pypi.org/project/ddmtolab/1.1.0/
```

### 3.6 验证发布

```bash
# 等待 1-2 分钟让 PyPI 索引更新

# 在新环境测试安装
conda create -n test_new python=3.10
conda activate test_new
pip install ddmtolab

# 检查版本
python -c "import ddmtolab; print(ddmtolab.__version__)"
# 应该输出: 1.1.0
```

---

## 4. 创建 GitHub Release

### 4.1 创建 Git 标签

```bash
# 创建标签
git tag -a v1.1.0 -m "Release version 1.1.0"

# 查看标签
git tag -l

# 推送标签到 GitHub
git push origin v1.1.0
```

### 4.2 在 GitHub 创建 Release

1. 访问：https://github.com/JiangtaoShen/DDMTOLab/releases/new

2. **选择标签**: `v1.1.0`

3. **Release 标题**: `v1.1.0 - [简短描述]`
   - 例如: `v1.1.0 - New Algorithms and Performance Improvements`

4. **描述内容**（模板）:

```markdown
## What's New in v1.1.0

### 🚀 New Features
- Added new PSO variants (SL-PSO, KL-PSO)
- Implemented distributed computing support
- New benchmark problems: CEC22-MTSO

### 🐛 Bug Fixes
- Fixed memory leak in MTBO
- Resolved convergence issues in NSGA-III
- Fixed documentation rendering errors

### 📈 Improvements
- 30% faster convergence for large-scale problems
- Reduced memory usage by 20%
- Improved error messages

### 📚 Documentation
- Updated installation guide
- Added new tutorials
- Improved API documentation

### 🔗 Links
- PyPI: https://pypi.org/project/ddmtolab/1.1.0/
- Documentation: https://jiangtaoshen.github.io/DDMTOLab/

### 📦 Installation

```bash
pip install ddmtolab==1.1.0
```

or upgrade:

```bash
pip install --upgrade ddmtolab
```

**Full Changelog**: https://github.com/JiangtaoShen/DDMTOLab/compare/v1.0.0...v1.1.0
```

5. **附加文件**（可选）:
   - 可以上传编译好的包（从 `dist/` 目录）
   - 不是必需的，用户可以从 PyPI 安装

6. **预发布**（Pre-release）:
   - 如果是测试版本（如 1.1.0-beta），勾选 "This is a pre-release"

7. 点击 **"Publish release"**

### 4.3 使用 GitHub CLI（可选，更快）

```bash
# 安装 GitHub CLI
# 从 https://cli.github.com/ 下载安装

# 登录
gh auth login

# 创建 release
gh release create v1.1.0 \
  --title "v1.1.0 - New Algorithms and Performance Improvements" \
  --notes-file CHANGELOG.md \
  dist/*
```

---

## 5. 常见问题

### Q1: 如果版本号忘记更新怎么办？

**不能修改已发布的版本！** 

正确做法：
```bash
# 1. 更新版本号（比如从 1.1.0 改到 1.1.1）
# 2. 重新构建和发布
python -m build
twine upload dist/*
```

### Q2: 如何撤回已发布的版本？

**PyPI 不允许删除版本！**

但可以 "yank"（标记为不推荐）：
```bash
# 在 PyPI 网页上操作
# 或使用 twine（需要 twine 5.0+）
twine upload --skip-existing dist/*
```

### Q3: 发布到 PyPI 后多久可以安装？

通常 1-2 分钟，最多 5 分钟。

### Q4: 如何发布预发布版本？

```bash
# 版本号使用 alpha/beta/rc 后缀
__version__ = "1.1.0-beta.1"
version = "1.1.0b1"  # PyPI 格式

# 构建上传
python -m build
twine upload dist/*

# 用户安装时需要指定
pip install ddmtolab==1.1.0b1
# 或安装最新的预发布版
pip install --pre ddmtolab
```

### Q5: 构建失败怎么办？

```bash
# 检查 pyproject.toml 语法
python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"

# 查看详细构建日志
python -m build --verbose

# 清理后重试
rmdir /s /q build dist *.egg-info
python -m build
```

### Q6: 如何只更新文档而不发布新版本？

```bash
# 更新文档
git add docs/
git commit -m "docs: update tutorial"
git push origin main

# 不需要发布新 PyPI 版本
# 只在有代码功能变化时才发布新版本
```

---

## 📋 快速发布检查清单

### 发布前检查
- [ ] 代码已测试通过
- [ ] 版本号已更新（3个文件）
- [ ] CHANGELOG 已更新
- [ ] 文档已更新
- [ ] 本地测试安装成功

### 发布流程
- [ ] Git 提交并推送
- [ ] 清理旧构建
- [ ] `python -m build`
- [ ] `twine check dist/*`
- [ ] `twine upload dist/*` (可先 testpypi)
- [ ] 验证 PyPI 安装
- [ ] 创建 Git 标签
- [ ] 创建 GitHub Release

### 发布后
- [ ] 在新环境测试安装
- [ ] 更新文档网站（如果有）
- [ ] 发布公告（Twitter/论坛等）
- [ ] 回复 Issue/PR

---

## 🔄 完整命令速查（复制粘贴用）

```bash
# ===== 1. 更新版本号 =====
# 手动编辑这 3 个文件:
# - src/ddmtolab/__init__.py
# - pyproject.toml  
# - conda/meta.yaml

# ===== 2. 本地测试 =====
conda activate ddmtolab
cd D:\DDMTOLab
pip install -e .
python -c "import ddmtolab; print(ddmtolab.__version__)"

# ===== 3. Git 提交 =====
git add .
git commit -m "release: bump version to 1.1.0"
git push origin main

# ===== 4. 构建发布 =====
rmdir /s /q build
rmdir /s /q dist
python -m build
twine check dist/*
twine upload dist/*

# ===== 5. 创建 Release =====
git tag -a v1.1.0 -m "Release version 1.1.0"
git push origin v1.1.0
# 然后访问 GitHub 网页创建 Release

# ===== 6. 验证 =====
# 等待 2 分钟
conda create -n test_v1.1.0 python=3.10
conda activate test_v1.1.0
pip install ddmtolab
python -c "import ddmtolab; print(ddmtolab.__version__)"
```

---

## 🎯 最佳实践建议

1. **版本发布频率**
   - Patch (1.0.x): 每 1-2 周，bug 修复
   - Minor (1.x.0): 每 1-3 月，新功能
   - Major (x.0.0): 每年或重大更新

2. **版本号规划**
   - 保持一致性
   - 遵循语义化版本
   - 记录在 CHANGELOG

3. **测试**
   - 每次发布前充分测试
   - 使用 TestPyPI 预发布
   - 在新环境验证安装

4. **文档**
   - 保持 CHANGELOG 更新
   - Release Notes 写清楚
   - 及时更新文档网站

5. **沟通**
   - 在 GitHub Discussions 发布公告
   - 通知主要用户
   - 处理反馈的 Issues

---

**保存这份文档**，以后每次更新就按照这个流程走！🚀

有问题随时问我！
