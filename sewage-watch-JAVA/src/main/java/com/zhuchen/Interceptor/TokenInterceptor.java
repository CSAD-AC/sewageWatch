package com.zhuchen.Interceptor;

import com.zhuchen.Utils.JwtUtil;
import io.jsonwebtoken.Claims;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.lang.Nullable;
import org.springframework.stereotype.Component;
import org.springframework.util.AntPathMatcher;
import org.springframework.web.servlet.HandlerInterceptor;
import org.springframework.web.servlet.ModelAndView;

import java.util.HashMap;
import java.util.List;
import java.util.Map;


@Slf4j
@Component
public class TokenInterceptor implements HandlerInterceptor {
    @Override // 目标资源方法执行前执行，返回值表示是否放行
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
        // 放行 OPTIONS 预检请求
        if ("OPTIONS".equalsIgnoreCase(request.getMethod())) {
            return true;
        }

        log.info("请求被拦截，正在验证token是否有足够权限");
        log.info("尝试访问{}", request.getRequestURI());
        String token = request.getHeader("Authorization");
        if (token != null && token.startsWith("Bearer ")) {
            String actualToken = token.substring(7);
            Map<String, List<String>> roleMapper = Map.ofEntries(
                Map.entry("PUBLIC",
                        List.of("/analysis", "/history", "/detection")),
                Map.entry("INSPECTOR",
                        List.of("/analysis", "/history", "/detection", "/tasks", "/tasks/*", "/monitor")),
                Map.entry("ADMIN",
                        List.of("/analysis", "/history", "/detection", "/tasks", "/tasks/*", "/monitor", "/admin/*"))
            );
            AntPathMatcher pathMatcher = new AntPathMatcher();
            // 验证 token 合法性
            if (isValidToken(actualToken)) {
                Claims claims = JwtUtil.parseToken(actualToken);
                // 获取角色权限等级
                String role = claims.get("role").toString();
                if (roleMapper.containsKey(role)) {
                    for (String pattern : roleMapper.get(role)) {
                        if (pathMatcher.match(pattern, request.getRequestURI())) {
                            log.info("权限等级{}尝试访问{}   已放行", role, request.getRequestURI());
                            return true;
                        }
                    }
                    response.sendError(HttpServletResponse.SC_FORBIDDEN, "无权限访问");
                    return false;
                }
            }
        }
        // 验证失败，返回 401
        log.info("请求被拦截，未放行");
        response.sendError(HttpServletResponse.SC_UNAUTHORIZED, "登录失效");
        // 验证失败，不放行
        return false;
    }
    private boolean isValidToken(String token) {
        return JwtUtil.validateToken(token);
    }
    @Override // 目标资源方法执行后执行
    public void postHandle(HttpServletRequest request, HttpServletResponse response, Object handler, @Nullable ModelAndView modelAndView) throws Exception {

    }
    @Override // 视图渲染完成之后执行，最后执行
    public void afterCompletion(HttpServletRequest request, HttpServletResponse response, Object handler, @Nullable Exception ex) throws Exception {

    }
}
