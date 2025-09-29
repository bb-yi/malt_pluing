#include "Common.glsl"

#define iMouse vec4(100., 100., 100.,100.)

/*  META
    @normal: default=NORMAL;
*/
vec3 random_normal_offset(vec3 normal, float angle, float seed)
{
    vec3 random_vec = random_per_pixel(seed).xyz;
    random_vec.xyz = random_vec.xyz * 2.0 - 1.0;

    vec3 tangent = normalize(random_vec - normal * dot(random_vec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);

    angle = random_per_pixel(seed).w * angle;
    return TBN * vec3(0, sin(angle), cos(angle));
}

/*  META
    @normal: default=NORMAL;
    @roughness: default=1.0;
    @samples: default=32;
*/
vec3 hdri_diffuse(sampler2D hdri, vec3 normal, float roughness, int samples)
{
    vec3 result = vec3(0);

    for(int i = 0; i < samples; i++)
    {
        vec2 uv = hdri_uv(random_normal_offset(normal, roughness*PI/2, i));
        result += texture(hdri, uv).rgb / samples;
    }
    return result;
}

/*  META
    @normal: default=NORMAL;
    @samples: default=32;
*/
vec3 hdri_specular(sampler2D hdri, vec3 normal, float roughness, int samples)
{
    normal = reflect(view_direction(), normal);
    return hdri_diffuse(hdri, normal, roughness, samples);
}

//https://www.shadertoy.com/view/XlfGRj
void star(out vec4 fragColor, in vec2 uv, in vec2 a, in vec2 move, in vec3 from, in uvec2 LoopCount)
{
    float formuparam = 0.53;
    float stepsize = 0.1;
    float tile = 0.850;
    float brightness = 0.0015;
    float darkmatter = 0.300;
    float distfading = 0.730;
    float saturation = 0.850;


    vec3 dir = vec3(uv, 1.);
    //mouse rotation
    float a1 = a.x;
    float a2 = a.y;
    mat2 rot1 = mat2(cos(a1), sin(a1), -sin(a1), cos(a1));
    mat2 rot2 = mat2(cos(a2), sin(a2), -sin(a2), cos(a2));
    dir.xz *= rot1;
    dir.xy *= rot2;
    // vec3 from = vec3(1., .5, 0.5);
    from += vec3(move, -2.);
    from.xz *= rot1;
    from.xy *= rot2;

    //volumetric rendering
    float s = 0.1, fade = 1.;
    vec3 v = vec3(0.);
    for (int r = 0; r < LoopCount.x; r++)
    {
        vec3 p = from + s * dir * .5;
        p = abs(vec3(tile) - mod(p, vec3(tile * 2.))); // tiling fold
        float pa, a = pa = 0.;
        for (int i = 0; i < LoopCount.y; i++)
        {
            p = abs(p) / dot(p, p) - formuparam; // the magic formula
            // p = abs(p) / max(dot(p, p), 0.0016) - formuparam; // the magic formula
            a += abs(length(p) - pa); // absolute sum of average change
            pa = length(p);
        }
        float dm = max(0., darkmatter - a * a * .001); //dark matter
        a *= a * a; // add contrast
        if (r > 6) fade *= 1. - dm; // dark matter, don't render near
        //v+=vec3(dm,dm*.5,0.);
        v += fade;
        v += vec3(s, s * s, s * s * s * s) * a * brightness * fade; // coloring based on distance
        fade *= distfading; // distance fading
        s += stepsize;
    }
    v = mix(vec3(length(v)), v, saturation); //color adjust
    fragColor = vec4(v * .01, 1.);
    // fragColor = vec4(UV[0].x, UV[0].y, 0., 1.);
}


//https://www.shadertoy.com/view/4d2Xzw
void BokehWithDepth(
    out vec4 fragColor1,
    in sampler2D tex,
    in sampler2D depthTex,        // 新增深度贴图
    in vec2 uv,
    in float radius,
    in int LoopCount,
    in float Y_scale,
    in float angle,
    in float depthThreshold
)
{
    // 旋转矩阵（黄金角旋转）
    const float goloen_angle = 2.3999632;
    mat2 rotGolden = mat2(
        cos(goloen_angle), sin(goloen_angle),
        -sin(goloen_angle), cos(goloen_angle)
    );

    vec3 acc = vec3(0.0);
    vec3 divWeight = vec3(0.0);

    // 当前像素的深度
    float depth_center = texture(depthTex, uv).r;

    float r = 1.0;
    vec2 vangle = vec2(0.0, radius * 0.01 / sqrt(float(LoopCount)));
    // 如果有发入 angle 参数，则先旋转这个初始方向
    float angleRad = angle;  // 你可以把 angle 转成你想要的单位
    vangle = vangle * mat2(cos(angleRad), sin(angleRad), -sin(angleRad), cos(angleRad));
    for (int j = 0; j < LoopCount; j++)
    {
        // 控制采样半径增长
        r += 1.0 / r;
        vangle = rotGolden * vangle;

        vec2 offset = (r - 1.0) * vangle * vec2(1.0, Y_scale);
        vec2 uv_s = uv + offset;

        // 先检查 uv_s 是否越界（可选，防止采样越出图像）
        uv_s = clamp(uv_s, vec2(0.0), vec2(1.0));

        vec3 col = texture(tex, uv_s).xyz;
        vec3 bokeh = pow(col, vec3(1.0));

        // 深度采样
        float depth_s = texture(depthTex, uv_s).r;
        float dDiff = abs(depth_s - depth_center);

        // 权重：如果深度差太大，就降低权重
        float w = 1.0;
        if (dDiff > depthThreshold)
        {
            // 可选：直接忽略 (w = 0)，或给很小权重
            w = exp(-(dDiff - depthThreshold) * 50.0); // 这个“50.0”可以调，越大边界越硬
        }
        acc += col * bokeh * w;
        divWeight += bokeh * w;
    }
    divWeight += 0.00001; // 避免除零错误
    vec3 result = acc / divWeight;
    fragColor1 = vec4(result, 1.0);
}
// 鱼眼映射函数：输入一个 sampler2D，对非正方形图像正确处理中心与边缘
// 参数：
//   tex        — 源图像
//   uv         — 当前像素的 UV 坐标 [0,1]×[0,1]
//   resolution — 图像分辨率 (宽,高)
//   strength   — 畸变强度标度（0 表示无畸变）
//   maxAngle   — 最大视场角度（弧度），决定 r=1 映射到视线角度
//   center     — 鱼眼中心 UV 坐标（通常 (0.5,0.5)）
//   radiusUV   — 鱼眼圆盘在 UV 空间的半径（相对于 center）
//   mode       — 边缘处理方式：0 = 普通 clamped 样本外返回原图，1 = 黑边，2 = 混合边缘
void fisheye(
    out vec4 fragColor,
    out vec2 uv_out,
    sampler2D tex,
    vec2 uv,
    vec2 resolution,
    float strength,
    float maxAngle,
    vec2 center,
    float radiusUV,
    int mode
)
{
    const float pi = 3.14159265358979323846;
    // 宽高比，用于 x 方向缩放
    float aspect = resolution.x / resolution.y;
    // 将 uv 移动坐标，以 center 为原点
    vec2 coord = uv - center;
    // 考虑宽高比，使圆盘在屏幕上不拉伸
    coord.x *= aspect;
    // 将它标准化到 radiusUV 对应的 r==1
    coord /= radiusUV;
    float r = length(coord);
    // 判断是否超出圆盘
    if (r > 1.0)
    {
        // 根据 mode 处理
        if (mode == 1)
        {
            // 黑边
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        }
        else if (mode == 2)
        {
            // 混合边缘，比如线性混合边界
            // 可渐变混合
            float fade = 1.0 - clamp((r - 1.0) / 0.05, 0.0, 1.0);
            vec4 colOrig = texture(tex, uv);
            fragColor = mix(vec4(0.0), colOrig, fade);
        }
        else
        {
            // 默认：采原图
            fragColor = texture(tex, uv);
        }
        uv_out = uv;
        return;
    }
    // r 在 [0,1] 内，做畸变映射
    // Equidistant 鱼眼映射
    float theta = r * maxAngle * strength;
    float phi = atan(coord.y, coord.x);
    // 构造方向向量
    float sinT = sin(theta);
    float cosT = cos(theta);
    vec3 dir = vec3(sinT * cos(phi),
        sinT * sin(phi),
        cosT);
    // 投射到 image plane z=1
    vec2 proj = dir.xy / dir.z;
    // 反向校正 x 方向的 aspect
    proj.x /= aspect;
    // 将 proj 映射回 UV，以 maxAngle 控制边缘缩放
    float denom = tan(maxAngle * 0.5);
    vec2 uv_src = center + proj * (radiusUV * 0.5 / denom);
    uv_src = clamp(uv_src, vec2(0.0), vec2(1.0));
    uv_out = uv_src;
    fragColor = texture(tex, uv_src);
}


#ifdef IS_MESH_SHADER
/*  META
    @exclude_light_type: default=0;
    @normal: default=NORMAL;
*/
vec3 pbr_custom(vec3 albedo, float metalness, float roughness,vec3 normal,int exclude_light_type)
{
    vec3 result = vec3(0);
    for (int i = 0; i < LIGHTS.lights_count; i++)
    {
        ivec4 light_group = ivec4(LIGHT_GROUP_INDEX(i));
        if(!any(equal(MATERIAL_LIGHT_GROUPS, light_group))) continue;
        
        Light L = LIGHTS.lights[i];
        if(L.type == exclude_light_type) continue;
        LitSurface LS = npr_lit_surface(POSITION, normal, ID.x, L, i, Settings.Receive_Shadow, Settings.Self_Shadow);

        float NoL = LS.NoL;
        NoL = max(MIN_DOT, NoL);

        float NoV = dot(LS.N, LS.V);
        NoV = max(MIN_DOT, NoV);

        float NoH = dot(LS.N, LS.H);
        NoH = max(MIN_DOT, NoH);

        float LoV = dot(LS.L, LS.V);
        LoV = max(MIN_DOT, LoV);

        float VoH = dot(LS.V, LS.H);
        VoH = max(MIN_DOT, VoH);

        float a = max(0.001, roughness * roughness);

        //Diffuse Models
        float burley = BRDF_burley(NoL, NoV, VoH, a);
        float oren_nayar = BRDF_oren_nayar(NoL, NoV, LoV, a);
        float lambert = BRDF_lambert(NoL);

        //Specular Distribution Models
        float d_phong = D_blinn_phong(NoH, a);
        float d_beckmann = D_beckmann(NoH, a);
        float d_ggx = D_GGX(NoH, a);

        //Specular Geometric Shadowing Models
        float g_cook_torrance = G_cook_torrance(NoH, NoV, NoL, VoH);
        float g_beckmann = G_beckmann(NoL, NoV, a);
        float g_ggx = G_GGX(NoL, NoV, a);

        float dielectric = 0.04;
        float F0 = mix(dielectric, 1.0, metalness);
        //Specular Fresnel Models
        float f_schlick = F_schlick(VoH, F0, 1.0);
        float f_cook_torrance = F_cook_torrance(VoH, F0);

        // Disney-like PBR shader (Burley for diffuse + GGX for speculars)
        vec3 diffuse_color = albedo * burley * (1.0 - f_schlick) * (1.0 - metalness);
        
        float specular = BRDF_specular_cook_torrance(d_ggx, f_schlick, g_ggx, NoL, NoV);
        vec3 specular_color = mix(vec3(specular), albedo * specular, metalness);

        vec3 lit_color = diffuse_color + specular_color;

        result += lit_color * LS.light_color * LS.shadow_multiply;
    }

    return result;
}

#endif


#ifdef IS_MESH_SHADER
/*  META
    @exclude_light_type: default=0;
    @normal: default=NORMAL;
*/
vec3 light_color(vec3 normal,int exclude_light_type)
{
    bool shadow;
    bool self_shadow;
    ivec4 light_groups = ivec4(1,0,0,0);
    _shadow_params(0, shadow, self_shadow);
    vec3 result = vec3(0,0,0);
    for (int i = 0; i < LIGHTS.lights_count; i++)
    {
        for(int group_index = 0; group_index < 4; group_index++)
        {
            if(LIGHT_GROUP_INDEX(i) != light_groups[group_index])
            {
                continue;
            }
            Light L = LIGHTS.lights[i];
            if(exclude_light_type == L.type)
            {
                continue;
            }
            LitSurface LS = npr_lit_surface(POSITION, normal, ID.x, L, i, shadow, self_shadow);
            if(LS.NoL < 0)
            {
                continue;
            }
            float lambert = LS.NoL;
            vec3 diffuse = lambert*LS.light_color * LS.shadow_multiply;
            result += diffuse;
        }
    }
    return result;
}

#endif