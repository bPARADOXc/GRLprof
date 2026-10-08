# GitHub 发布清单

## 当前结论

本项目可按**研究源码发布**准备，不等同于已完成“全新机器克隆即复现”的完整发行版。
本地真实数据短测已验证两个阶段的训练与推理；完整训练、公开资产分发和全新环境安装尚未验证。

本轮发布检查：151 个候选源码文件常见敏感模式扫描无错误；99 个 Python 文件 AST、
9 个 Shell 脚本语法及文档本地链接检查通过，5 项发布工具测试通过。
训练原生入口 `--help` 和短测脚本 dry-run 通过；GitHub Actions 配置已添加，
尚未在远程运行。这些检查不替代上一轮的 GPU 短测或未来的全新环境验证。

已处理：

- README 以独立的 GRLprof 方法介绍，第三方来源集中保留在 NOTICE 和许可证中。
- 文档中的机器用户名、私人挂载路径已替换为可移植示例。
- 数据、权重、日志、缓存、私有环境配置不进入源码包。
- 实测依赖版本与历史 requirements 分开，明确验证边界。
- 添加可重复的发布扫描、导出工具、CPU 测试和 GitHub Actions 源码检查。
- 用户选择无旧历史发布；原工作仓库及其 Git 历史不修改。

待正式复现发行版补齐：

1. 确认并提供非文本实体 ID 表、评测别名资源的来源及允许分发方式。
2. 决定是否发布检索器 / 映射器 checkpoint；若发布，补下载地址、哈希及适用数据配置。
3. 在全新环境按照 README 安装并执行短测，必要时提供经过验证的完整锁文件。
4. 填写最终论文引用、真实项目仓库地址。不要编造作者、DOI 或发布日期。

## 本地检查与快照

```bash
python tools/check_release.py
python -m unittest discover -s tests -v
git diff --check
python tools/export_source.py --output outputs/release/GRLprof-source.zip
```

导出器选择当前 Git 已跟踪文件及非忽略的新文件，先检查再打包。
即使大文件被误加入 Git、后来又写入 ignore，也仍会被检查并阻止。
不会包含 `.git`、被忽略的数据、缓存和日志；保留源码及脚本执行权限。
目标压缩包已存在时拒绝覆盖，请选择新文件名。

当前工作仓库仍有历史作者个人邮箱，因此默认历史扫描会提示警告。
这是保留原仓库的预期状态，不应直接推送该历史。源码快照扫描不读取历史，
不会因为旧作者邮箱把历史带进包中。自动规则只能检查常见模式，仍需人工检查文件清单。

## 用 noreply 邮箱建立全新发布仓库

解压源码包到一个新目录；**不要复制原仓库的 `.git`，不要覆盖当前工作目录**。
在 GitHub 邮箱设置中复制自己的准确 noreply 邮箱；通常包含账户 ID 和用户名，
不要照抄下面占位值。首次 commit 之前配置：

```bash
# 在解压得到的全新 GRLprof 目录内执行，而不是原工作仓库
git init
git config user.name "YOUR_PUBLIC_DISPLAY_NAME"
git config user.email "YOUR_EXACT_GITHUB_NOREPLY_EMAIL"
python tools/check_release.py --no-history
git add .
git diff --cached --stat
git diff --cached --check
git commit -m "Initial public source release"
```

上述 Git 命令是供维护者确认后执行的步骤，本次检查没有自动执行提交或推送。
新提交中的显示名也会公开；如需要匿名发布，请自行选择非实名显示名。

不要将未脱敏的 `pip freeze`、环境导出文件、终端日志、训练 config 或含本机路径的
checkpoint 一并公开。公开模型权重属于单独的发布步骤，需要另行检查其配置元数据。
