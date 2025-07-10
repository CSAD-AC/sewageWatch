CREATE DATABASE IF NOT EXISTS sewagewatch;

USE sewagewatch;

CREATE TABLE IF NOT EXISTS user (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '用户ID',
    userName  VARCHAR(20) UNIQUE NOT NULL COMMENT '用户名',
    password VARCHAR(80) NOT NULL COMMENT 'BCrypt加密密码(60位)',
    name VARCHAR(20) NOT NULL COMMENT '用户昵称',
    role VARCHAR(20) NOT NULL COMMENT '用户角色权限',
    enabled BOOLEAN NOT NULL DEFAULT TRUE COMMENT '用户是否可用',
    updateTime DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT '用户表';


CREATE TABLE IF NOT EXISTS task (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '任务ID',
    title VARCHAR(255) NOT NULL COMMENT '任务标题',
    description TEXT COMMENT '任务描述',
    status VARCHAR(20) NOT NULL COMMENT '任务状态: TODO, DOING, DONE',
    priority VARCHAR(20) NOT NULL COMMENT '任务优先级: LOWEST, LOW, MEDIUM, HIGH, HIGHEST',
    deadline DATETIME COMMENT '任务截止时间',
    createdTime DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '任务创建时间',
    updatedTime DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '任务最后更新时间'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='任务表';

# 密码均为123456
INSERT INTO user (id, userName, password, name, role, enabled, updateTime)
VALUES
    (1, '3029489598', '$2a$10$8Jzu4r3XO1IezN8aMJ.ZGeI0QPg3DY/ed9KNKnU9djkmjtDLcY9Iy', '逐辰', 'INSPECTOR', true, '2025-06-22 11:02:35'),
    (2, '123456', '$2a$10$HJusN5zCTdWrz3kq5sz4ROSlyi7WVZJyerwv7./3yNQmUviov7Dlm', '测试', 'PUBLIC', true, '2025-06-22 11:47:20');
    (3, 'ADMIN', '$2a$10$8Jzu4r3XO1IezN8aMJ.ZGeI0QPg3DY/ed9KNKnU9djkmjtDLcY9Iy', '管理员', 'ADMIN', true, '2025-06-21 13:30:25');

INSERT INTO task (id, title, description, status, priority, deadline, createdTime, updatedTime)
VALUES
    (1, '日常巡逻', NULL, 'TODO', 'LOW',
     '2026-06-14 19:51:56', '2025-06-14 19:51:55', '2025-06-14 19:52:01'),
    (2, '定点确认', '接到居民报告，左2河道处有污水，现派遣无人机前往确认情况',
     'TODO', 'HIGH', '2025-07-14 21:52:11',
     '2025-06-14 21:52:19', '2025-06-14 21:52:31');

# 测试数据
INSERT INTO task (title, description, status, priority, deadline) VALUES
    ('完成项目报告', '撰写季度项目进度报告', 'TODO', 'HIGH', '2025-07-15 17:00:00'),
    ('修复登录bug', '解决用户登录时的验证码问题', 'DOING', 'HIGHEST', '2025-06-20 12:00:00'),
    ('设计新logo', '为公司设计新的品牌标识', 'TODO', 'MEDIUM', '2025-07-01 00:00:00'),
    ('客户会议准备', '准备下周与重要客户的演示材料', 'DOING', 'HIGH', '2025-06-25 09:00:00'),
    ('代码审查', '审查团队成员的pull request', 'DONE', 'MEDIUM', '2025-06-15 15:00:00'),
    ('更新文档', '更新API接口文档', 'TODO', 'LOW', '2025-06-30 18:00:00'),
    ('服务器维护', '执行月度服务器维护任务', 'DOING', 'HIGHEST', '2025-06-21 02:00:00'),
    ('学习新技术', '学习新的前端框架', 'TODO', 'LOWEST', '2025-07-10 00:00:00'),
    ('团队建设活动', '组织部门团建活动', 'TODO', 'MEDIUM', '2025-07-05 00:00:00'),
    ('性能优化', '优化数据库查询性能', 'DOING', 'HIGH', '2025-06-22 00:00:00'),
    ('招聘面试', '面试前端开发候选人', 'DONE', 'HIGH', '2025-06-18 14:00:00'),
    ('财务报告', '准备季度财务报告', 'TODO', 'HIGHEST', '2025-06-30 23:59:59'),
    ('用户调研', '进行新产品用户调研', 'TODO', 'MEDIUM', '2025-07-08 00:00:00'),
    ('测试新功能', '测试即将发布的新功能', 'DOING', 'HIGH', '2025-06-23 18:00:00'),
    ('更新简历', '更新个人简历', 'TODO', 'LOWEST', '2025-07-31 00:00:00'),
    ('安全审计', '执行系统安全审计', 'TODO', 'HIGHEST', '2025-07-12 00:00:00'),
    ('撰写博客', '写一篇技术博客文章', 'DONE', 'LOW', '2025-06-17 00:00:00'),
    ('产品演示', '为客户进行产品演示', 'DOING', 'HIGH', '2025-06-24 10:00:00'),
    ('清理邮箱', '清理和整理工作邮箱', 'TODO', 'LOW', '2025-06-28 00:00:00'),
    ('备份数据库', '执行数据库全量备份', 'DOING', 'HIGHEST', '2025-06-20 03:00:00'),
    ('设计UI界面', '设计新功能的用户界面', 'TODO', 'MEDIUM', '2025-07-03 00:00:00'),
    ('解决生产问题', '解决生产环境中的紧急问题', 'DOING', 'HIGHEST', '2025-06-19 15:00:00'),
    ('编写测试用例', '为新功能编写测试用例', 'TODO', 'MEDIUM', '2025-06-27 00:00:00'),
    ('部门会议', '参加部门周例会', 'DONE', 'LOW', '2025-06-17 10:00:00'),
    ('学习英语', '完成本周英语学习计划', 'DOING', 'LOWEST', '2025-06-21 20:00:00'),
    ('更新依赖', '更新项目依赖库版本', 'TODO', 'MEDIUM', '2025-06-26 00:00:00'),
    ('重构代码', '重构旧模块的代码', 'DOING', 'HIGH', '2025-06-25 00:00:00'),
    ('制定计划', '制定下季度工作计划', 'TODO', 'HIGH', '2025-06-30 00:00:00'),
    ('客户支持', '处理客户支持请求', 'DOING', 'MEDIUM', '2025-06-19 16:00:00'),
    ('健身计划', '完成本周健身计划', 'TODO', 'LOWEST', '2025-06-22 19:00:00'),
    ('部署新版本', '部署应用新版本到生产环境', 'DOING', 'HIGHEST', '2025-06-21 00:00:00'),
    ('市场分析', '分析竞争对手市场策略', 'TODO', 'MEDIUM', '2025-07-07 00:00:00'),
    ('修复UI问题', '解决用户报告的UI显示问题', 'DOING', 'HIGH', '2025-06-20 18:00:00'),
    ('读书', '阅读专业书籍一章', 'TODO', 'LOW', '2025-06-23 21:00:00'),
    ('系统监控', '检查系统监控告警', 'DONE', 'HIGH', '2025-06-18 09:00:00'),
    ('编写文档', '编写新API使用文档', 'DOING', 'MEDIUM', '2025-06-22 00:00:00'),
    ('团队培训', '为团队进行新技术培训', 'TODO', 'HIGH', '2025-07-02 14:00:00'),
    ('优化CI/CD', '优化持续集成流程', 'DOING', 'HIGH', '2025-06-24 00:00:00'),
    ('整理桌面', '整理办公桌面和文件', 'TODO', 'LOWEST', '2025-06-19 17:00:00'),
    ('安全更新', '应用最新的安全补丁', 'DOING', 'HIGHEST', '2025-06-20 00:00:00'),
    ('产品规划', '规划下季度产品路线图', 'TODO', 'HIGHEST', '2025-07-10 00:00:00'),
    ('解决冲突', '解决代码合并冲突', 'DOING', 'HIGH', '2025-06-19 11:00:00'),
    ('冥想', '完成每日冥想练习', 'TODO', 'LOWEST', '2025-06-20 07:00:00'),
    ('测试自动化', '编写自动化测试脚本', 'DOING', 'MEDIUM', '2025-06-23 00:00:00'),
    ('客户回访', '回访上月签约客户', 'TODO', 'MEDIUM', '2025-06-28 10:00:00'),
    ('升级服务器', '升级生产服务器硬件', 'DOING', 'HIGHEST', '2025-06-22 02:00:00'),
    ('学习算法', '完成算法练习题', 'TODO', 'LOW', '2025-06-25 20:00:00'),
    ('数据库迁移', '将数据迁移到新数据库', 'DOING', 'HIGHEST', '2025-06-23 00:00:00'),
    ('编写提案', '编写项目立项提案', 'TODO', 'HIGH', '2025-06-27 00:00:00'),
    ('解决性能问题', '解决系统性能瓶颈', 'DOING', 'HIGHEST', '2025-06-19 14:00:00'),
    ('学习设计模式', '学习并实践设计模式', 'TODO', 'LOW', '2025-06-26 20:00:00'),
    ('客户培训', '为客户进行产品使用培训', 'DOING', 'MEDIUM', '2025-06-24 15:00:00'),
    ('更新许可证', '更新软件许可证', 'TODO', 'HIGH', '2025-06-30 00:00:00'),
    ('修复安全漏洞', '修复发现的安全漏洞', 'DOING', 'HIGHEST', '2025-06-20 00:00:00'),
    ('编写周报', '编写本周工作总结', 'TODO', 'LOW', '2025-06-21 17:00:00'),
    ('优化搜索功能', '优化产品搜索功能', 'DOING', 'HIGH', '2025-06-23 00:00:00'),
    ('学习管理技巧', '阅读管理类书籍', 'TODO', 'LOWEST', '2025-07-15 00:00:00'),
    ('解决兼容性问题', '解决浏览器兼容性问题', 'DOING', 'HIGH', '2025-06-22 18:00:00'),
    ('准备演讲', '准备技术大会演讲材料', 'TODO', 'MEDIUM', '2025-07-05 00:00:00'),
    ('更新简历', '更新个人工作经历', 'DONE', 'LOWEST', '2025-06-18 00:00:00'),
    ('优化缓存策略', '优化应用缓存策略', 'DOING', 'HIGH', '2025-06-24 00:00:00'),
    ('学习数据分析', '学习数据分析技术', 'TODO', 'LOW', '2025-07-08 20:00:00'),
    ('解决客户投诉', '处理客户投诉问题', 'DOING', 'HIGHEST', '2025-06-19 10:00:00'),
    ('编写技术文档', '编写系统架构文档', 'TODO', 'MEDIUM', '2025-06-29 00:00:00'),
    ('优化登录流程', '优化用户登录体验', 'DOING', 'HIGH', '2025-06-22 00:00:00'),
    ('学习云技术', '学习云计算平台使用', 'TODO', 'LOW', '2025-07-12 00:00:00'),
    ('解决数据库死锁', '解决生产环境数据库死锁', 'DOING', 'HIGHEST', '2025-06-20 16:00:00'),
    ('编写用户手册', '编写产品用户手册', 'TODO', 'MEDIUM', '2025-07-01 00:00:00'),
    ('优化响应速度', '优化页面加载速度', 'DOING', 'HIGH', '2025-06-23 00:00:00'),
    ('学习新语言', '学习一门新的编程语言', 'TODO', 'LOWEST', '2025-07-31 00:00:00'),
    ('解决内存泄漏', '解决应用内存泄漏问题', 'DOING', 'HIGHEST', '2025-06-21 14:00:00'),
    ('编写测试报告', '编写系统测试报告', 'TODO', 'MEDIUM', '2025-06-28 00:00:00'),
    ('优化用户界面', '优化产品用户界面', 'DOING', 'HIGH', '2025-06-24 00:00:00'),
    ('学习人工智能', '学习AI基础知识', 'TODO', 'LOW', '2025-07-15 00:00:00'),
    ('解决网络问题', '解决办公室网络故障', 'DOING', 'HIGHEST', '2025-06-20 11:00:00'),
    ('编写项目总结', '编写项目结项报告', 'TODO', 'HIGH', '2025-06-30 00:00:00');











