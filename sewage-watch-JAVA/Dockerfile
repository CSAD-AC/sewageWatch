# 构建阶段
FROM maven:3.9.9-ibm-semeru-17-noble AS build
WORKDIR /app

# 分阶段下载依赖（利用Docker缓存）
COPY pom.xml .
RUN sed -i 's#https://repo.maven.apache.org#https://maven.aliyun.com/repository/public#g' /usr/share/maven/conf/settings.xml
RUN mvn dependency:go-offline -B

# 构建应用
COPY src ./src
RUN mvn clean package -DskipTests

# 运行阶段
FROM eclipse-temurin:17-jdk-alpine
WORKDIR /app

# 复制构建产物
COPY --from=build --chown=appuser /app/target/*.jar app.jar

EXPOSE 8080
ENTRYPOINT ["java", "-jar", "/app/app.jar"]