#define PI 3.1415926535897932384626433832795
/* META
    @label: Smoothstep;
    @edge0: subtype=Slider;default=0.0;min=0.0;max=1.0;
    @edge1: subtype=Slider;default=1.0;min=0.0;max=1.0;
*/
void my_smoothstep( in float x,in float edge0, in float edge1, out float result) {
    result = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    result = result * result * (3.0 - 2.0 * result);
}
//三种颜色的渐变
/*  META
    @position1: subtype=Slider;default=0.0;min=0.0;max=1.0;
    @position2: subtype=Slider;default=0.5;min=0.0;max=1.0;
    @position3: subtype=Slider;default=1.0;min=0.0;max=1.0;
    @color1: default=(0.0, 0.0, 0.0);
    @color2: default=(0.5, 0.5, 0.5);
    @color3: default=(1.0, 1.0, 1.0);
*/
void ramp_3colors(in float x,in float position1,in vec3 color1, 
                    in float position2,in vec3 color2, 
                    in float position3,in vec3 color3,
                    out vec3 result)
{
    if (x <= position1) {
        result = color1;
    }
    else if (x >= position3) {
        result = color3;
    }
    else if (x <= position2) {
        // 在 [position1, position2] 区间进行 color1 → color2 插值
        float t = (x - position1) / (position2 - position1);
        result = mix(color1, color2, t);
    }
    else {
        // 在 [position2, position3] 区间进行 color2 → color3 插值
        float t = (x - position2) / (position3 - position2);
        result = mix(color2, color3, t);
    }
}
void DirToSDFAngle(in vec3 face_dir,in vec3 sun_dir,out float angle)
{
    face_dir = normalize(face_dir*vec3(1,1,0));
    sun_dir = normalize(sun_dir*vec3(1,1,0));
    float face_angle = atan(face_dir.y, face_dir.x);
    float sun_angle = atan(sun_dir.y, sun_dir.x);
    sun_angle = mod(sun_angle-PI/2.0, 2.0*PI);
    face_angle = mod(face_angle+PI/2.0, 2.0*PI)-PI;
    angle= sun_angle-face_angle;
}

/*  META
    @UV: label=UV; default=UV[0];
    @smoothness: subtype=Slider;default=0.02;min=0.0;max=1.0;
*/
void toon_sdf_mask(in sampler2D SDF, in vec2 UV,in float angle,in float smoothness,in float offset, out float mask,out float SDF_R,out float SDF_G,out float SDF_B,out float SDF_A)
{
    vec4 SDF_RIGHT = texture(SDF, UV);
    float SDF_R_F =( SDF_RIGHT.r+SDF_RIGHT.b+SDF_RIGHT.a)/3.0+offset;
    vec4 SDF_LEFT = texture(SDF, vec2(1.0-UV.x, UV.y));
    float SDF_L_F =(SDF_LEFT.r+SDF_LEFT.b+SDF_LEFT.a)/3.0+offset;
    angle+=PI;
    angle =mod(angle+PI, 2.0*PI)-PI;
    float Threshold = -cos(angle)*0.5+0.5;
    if (angle>0.0)
    {
    mask = smoothstep(Threshold-smoothness,Threshold+smoothness,SDF_R_F);
    }
    else
    {
    mask = smoothstep(Threshold-smoothness,Threshold+smoothness,SDF_L_F);
    }
    SDF_R = SDF_RIGHT.r;
    SDF_G = SDF_RIGHT.g;
    SDF_B = SDF_RIGHT.b;
    SDF_A = SDF_RIGHT.a;
}

//采样绿色法线
/*  META
    @UV: label=UV; default=UV[0];
*/
void Smapler_green_normal(in sampler2D tex, in vec2 UV, out vec3 normal,out float tex_b,out float tex_a)
{
    vec4 tex_color = texture(tex, UV);
    normal = vec3(tex_color.r, tex_color.g, sqrt(1-pow(tex_color.r, 2)-pow(tex_color.g, 2)));
    tex_b = tex_color.b;
    tex_a = tex_color.a;
}

//法线缩放
/*  META
    @baseNormal: default=NORMAL;
*/
void NormalScale(in vec3 normal,in vec3 baseNormal,in float scale, out vec3 result,out vec3 baseNormal_out)
{
    result = normalize(normal * scale + baseNormal * (1.0 - scale));
    baseNormal_out = baseNormal;
}

void GetImageSize(in sampler2D Image, out vec2 size)
{
    size = vec2(textureSize(Image, 0));
}
void HasImage(in sampler2D Image, out bool result)
{
    result = textureSize(Image, 0).x > 1;
}
/* META @meta: label=My HSV Edit; */
vec4 my_hsv_edit(vec4 color, float mask,float hue, float saturation, float value)
{
    vec3 hsv = rgb_to_hsv(color.rgb);
    hsv += vec3(hue, saturation, value)*vec3(mask);
    return vec4(hsv_to_rgb(hsv), color.a);
}

//*颜色转整数 作为ID用
/* META
    @x: default=0;
*/
vec3 packIntToVec3(int x) {
    float fx = float(x);
    vec3 v;
    v.x = mod(fx, 256.0);     fx = floor(fx / 256.0);
    v.y = mod(fx, 256.0);     fx = floor(fx / 256.0);
    v.z = mod(fx, 256.0);

    return v / 255.0;  // 归一化到 [0,1]
}

/* META
    @v: default=(0.0, 0.0,0.0,0.0);
*/
int unpackVec3ToInt(vec3 v) {
    vec3 bytes = round(v * 255.0);  // 还原 0~255
    float value = bytes.x +
                  bytes.y * 256.0 +
                  bytes.z * 65536.0;
    return int(value);
}

//*位置转深度
/*  META
    @Position      : subtype=Vector; default=(0,0,0,0);
    @Projection    : subtype=Matrix; default=PROJECTION;
    @depth         : default=0.0;
    @normalizedDepth: default=0.0;
    @isBackground  : default=false;
    @isForeground  : default=true;
*/
void PositionToDepths(
    in  vec4 Position,           // 世界空间位置（w==0 表示背景）
    in  mat4 Projection,         // 相机投影矩阵
    out float depth,             // 相机空间线性深度（世界单位）
    out float normalizedDepth,   // 线性归一化深度 [0,1]
    out bool  isBackground,      // 背景标记
    out bool  isForeground       // 前景标记
){
    /* ---------- 1. 背景快速分支 ---------- */
    if (Position.w == 0.0)
    {
        depth           = 1000.0;   // 足够大的线性深度
        normalizedDepth = 1.0;
        isBackground    = true;
        isForeground    = false;
        return;
    }

    /* ---------- 2. 世界 → 相机空间 ---------- */
    vec3 worldPos = Position.xyz;
    Transform(0,      // Type = Point
              1,      // From = World
              2,      // To   = Camera
              worldPos);
    vec3 viewPos = worldPos;   // 此时已在相机空间

    /* ---------- 3. 线性深度（OpenGL 相机朝 -Z） ---------- */
    depth = -viewPos.z;

    /* ---------- 4. 线性归一化深度 [0,1] ---------- */
    float n = abs(Projection[3][2] / (Projection[2][2] - 1.0)); // 近平面
    float f = abs(Projection[3][2] / (Projection[2][2] + 1.0)); // 远平面
    normalizedDepth = clamp((depth - n) / (f - n), 0.0, 1.0);

    /* ---------- 5. 前景/背景标记 ---------- */
    isBackground = false;
    isForeground = true;
}


float _samplePosToDepth(in sampler2D p, in vec2 uv)
{
    float temp_float;
    bool temp_bool;
    vec3 pos = texture(p, uv).xyz;
    float depth;
    PositionToDepths(vec4(pos, 1.0), PROJECTION, depth, temp_float, temp_bool, temp_bool);
    return depth;
}


//* 屏幕空间曲率和边缘光 来自Goo引擎
/*  META
    @uv:default=screen_uv();
*/
void screen_rim(in sampler2D depth,
                in sampler2D position,
                in vec2 uv,
                in int n_samples,
                in float sample_scale,
                in float clamp_dist,
                out float curvature,
                out float rim,
                out float rim_inside,
                out vec4 pos,
                out float depth_out)
{
    bool use_position= false;
    HasImage(depth, use_position);
    use_position=!use_position;
    // 固定分辨率换成基于屏幕的 texel size
    vec2 texel_size = vec2(1.0 / 1920.0, 1.0 / 1080.0);
    // 当前像素深度
    float mid_depth = use_position?(texture(position, uv).a!=0.0?_samplePosToDepth(position, uv):1000.0):texture(depth, uv).a;
    pos= texture(position, uv);
    depth_out = mid_depth;
    float clamp_range = 0.001;
    float i_samples = (64.0 / float(n_samples));

    float accum = 0.0;
    float rim_accum = 0.0;
    float rim_inside_accum = 0.0;
    // 随机旋转偏移，避免条纹伪影
    float hash = fract(sin(dot(uv, vec2(12.9898,78.233))) * 43758.5453);

    for (int r = 0; r < 8; r++)
    {
        float angle = (float(r) + hash) * 3.14159265 * 0.125; // 22.5°
        vec2 offset = vec2(cos(angle), sin(angle)) * texel_size * sample_scale;

        for (int i = 1; i <= n_samples; i++)
        {
            float left = use_position?(texture(position, uv + offset * i * i_samples).a!=0.0?_samplePosToDepth(position, uv + offset * i * i_samples):1000.0):texture(depth, uv + offset * i * i_samples).a;
            float right =use_position?(texture(position, uv - offset * i * i_samples).a!=0.0?_samplePosToDepth(position, uv - offset * i * i_samples):1000.0):texture(depth, uv - offset * i * i_samples).a;

            float curve = clamp(left - mid_depth, -clamp_range, clamp_range) +
                          clamp(right - mid_depth, -clamp_range, clamp_range);

            float afac = (1.0 - float(i - 1) / float(n_samples));

            float ad = max(abs(max(left, mid_depth) - max(right, mid_depth)) - clamp_dist, 0.0);

            accum += curve * afac * 0.001;
            rim_accum += max(min(mid_depth - min(left, right), clamp_dist), 0.0) * afac + max(min(max(left, right) - mid_depth, clamp_dist), 0.0) * afac;
            rim_inside_accum += max(min(max(left, right) - mid_depth, clamp_dist), 0.0) * afac;
            // rim_accum += min(mid_depth - min(left, right), clamp_dist) * afac;
        }
    }
    curvature = -accum / length(texel_size) * i_samples;
    rim = rim_accum / sample_scale * clamp_range;
    rim_inside = rim_inside_accum / sample_scale * clamp_range;
}

// *按编号选择灯光
void Get_Light(in int light_index, out Light L)
{
    L = LIGHTS.lights[light_index];
}

#if defined(IS_MESH_SHADER) || defined(IS_SCREEN_SHADER)
#ifdef IS_MESH_SHADER
/*  META
    @position: subtype=Vector; default=(0.0,0.0,0.0);
    @color: default=(0.0,0.0,0.0);
    @strength: default=0.0;
    @type: default=0;
    @direction: default=(0.0,0.0,0.0);
    @spot_angle: default=0.0;
    @spot_blend: default=0.0;
    @light_groups: default=MATERIAL_LIGHT_GROUPS;
*/
#else
/*  META
    @position: subtype=Vector; default=(0.0,0.0,0.0);
    @color: default=(0.0,0.0,0.0);
    @strength: default=0.0;
    @type: default=0;
    @direction: default=(0.0,0.0,0.0);
    @spot_angle: default=0.0;
    @spot_blend: default=0.0;
    @light_groups: default=ivec4(1,0,0,0);
*/
#endif
void find_light(
    in vec3 position,
    in vec3 color,
    in float Strength,
    in int type,
    in vec3 direction,
    in float spot_angle,
    in float spot_blend,
    in ivec4 light_groups,
    out bool found,          // 是否找到
    out Light L              // 找到的灯光
    // *灯光过滤函数，根据灯光属性进行过滤，返回找到的第一个满足条件的灯光
)
{
    L.position     = vec3(0.0);
    L.direction    = vec3(0.0, 0.0, -1.0);
    L.color  = vec3(0.0);
    L.type         = 0;
    L.spot_angle   = 0.0;
    L.spot_blend   = 0.0;
    found = false;
    spot_angle = radians(spot_angle);
    spot_blend = radians(spot_blend);
    for (int i = 0; i < LIGHTS.lights_count; i++)
    {
        Light L_temp = LIGHTS.lights[i];
        // 灯光分组匹配
        bool group_match = false;
        for(int g = 0; g < 4; g++)
        {
            if(light_groups[g] == 0 || LIGHT_GROUP_INDEX(i) == light_groups[g])
            {
                group_match = true;
                break;
            }
        }
        if(!group_match) continue;
        //判断条件
        // 位置判断
        if(length(position) > 0.0 && distance(L_temp.position, position) > 0.001) continue;

        // 颜色判断
        if(length(color) > 0.0 && distance(L_temp.color, color) > 0.001) continue;

        float light_strength = max(max(L_temp.color.r, L_temp.color.g), L_temp.color.b);
        // 强度判断
        if(Strength != 0 && light_strength != Strength) continue;

        // 类型判断
        if(type != 0 && L_temp.type != type) continue;

        // 方向判断
        if(length(direction) > 0.0 && distance(L_temp.direction, direction) > 0.001) continue;

        // 聚光角度判断
        if(spot_angle > 0.0 && abs(L_temp.spot_angle - spot_angle) > 0.001) continue;

        // 聚光混合判断
        if(spot_blend > 0.0 && abs(L_temp.spot_blend - spot_blend) > 0.001) continue;

        // 找到第一个满足条件的灯光
        L = L_temp;
        L.direction=-L.direction;
        found = true;
        break;
    }
    
}
    #endif