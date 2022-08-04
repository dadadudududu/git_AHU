# Git 学习
## 概述
---
### 分类
- 集中式版本控制
- 分布式版本控制

### 工作机制
代码托管中心 == 远程库
- 局域网Gitlab
- GitHub 国外
- Gitee 国内
![流程图](2022-08-04-04-16-13.png)
## 安装
## 命令
---
设置用户签名
```git
git config --global user.name Oyster

git config --global user.email youni1994@163.com
```
初始化本地库
```
git init
```
查看本地库状态
```
git status
```
添加到暂存区，删除暂存区文件
```
git add file_name
git rm --cached file_name
```
提交到本地库
```
git commit -m"日志信息" 文件名
```
版本穿梭，git切换版本，底层其实是移动的head指针（指向master、branch）
```
git reset --hard 版本号
```
查看版本信息，引用日志信息、详细日志信息
```
git reflog
git log
```
## 分支操作
---
![服务器的一般部署](2022-08-04-05-05-25.png)
分支是在版本控制过程中，同时推进多个任务，为每个任务，我们就可以创建每个任务的单独分支，使用分支意味着程序员可以把自己的工作从开发主线分离开来，开发自己分支的时候，不会影响主线分支的运行。（分支底层是指针的引用）

创建分支
```
git branch branch_name
```
查看分支
```
git branch -v 
```
切换分支
```
git checkout branch_name
```
把指定的分支合并到当前分支上
```
git merge branch_name
```

**合并有冲突的分支**
冲突产生的原因：合并分支时，两个分支在同一个文件的同一个位置有两个完全不同的修改，git无法替我们决定使用哪一个，必须人为决定新代码内容。

手动统一版本文件之后执行add commit等一系列操作，不添加文件名（会报错）
```
git commit -m "版本信息"
```
## 团队协作机制
---
- 团队内协作
![](2022-08-04-06-21-23.png)
- 跨团队协作
![](2022-08-04-06-23-44.png)
## Github
查看当前所有远程地址别名
```
git remote -v
```
起别名
```
git remote add 别名 远程地址
```
推送本地分支上的内容到远程仓库
```
git push 别名 分支
```
将远程仓库的内容克隆到本地:
1. 拉取代码
2. 初始化本地库
3. 创建别名
```
git clone 远程地址
```
将远程仓库对于分支最新内容拉下来后与当前本地分支直接合并
```
git pull 远程地址别名 远程分支名
```

团队内协作对代码的拉取和推送，需要团队加入（manage access）

团队间协作 fork pullrequest

**SSH免密登陆**
1. -t rsa指定rsa加密算法,回车3次
```
ssh-keygen -t rsa -C youni1994@163.com
```
2. 复制公钥
```
cd .ssh
cat id_rsa.pub
```
3. github账号设置内粘贴公钥
