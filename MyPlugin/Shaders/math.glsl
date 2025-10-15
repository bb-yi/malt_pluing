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


/* META
@x: subtype=Float;label=X;default=0.0;
@inMin: subtype=Float;label=Input Min;default=0.0;
@inMax: subtype=Float;label=Input Max;default=1.0;
@outMin: subtype=Float;label=Output Min;default=0.0;
@outMax: subtype=Float;label=Output Max;default=1.0;
*/
float float_remap(float x, float inMin, float inMax, float outMin, float outMax)
{
    return outMin + ( (x - inMin) / (inMax - inMin) ) * (outMax - outMin);
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
    normal = vec3(tex_color.r, tex_color.g, sqrt(1-pow(tex_color.r*2.0-1.0, 2)-pow(tex_color.g*2.0-1.0, 2)));
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
    in float mask,
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
            
            accum += curve * afac * 0.001*mask;
            rim_accum += max(min(mid_depth - min(left, right), clamp_dist), 0.0) * afac*mask + max(min(max(left, right) - mid_depth, clamp_dist), 0.0) * afac*mask;
            rim_inside_accum += max(min(max(left, right) - mid_depth, clamp_dist), 0.0) * afac*mask;
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
    in float radius,
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
        
        // 类型判断 1:日光 2:点光 3:聚光
        if(type != 0 && L_temp.type != type) continue;
        
        // 方向判断
        if(length(direction) > 0.0 && distance(L_temp.direction, direction) > 0.001) continue;
        
        // 聚光角度判断
        if(spot_angle > 0.0 && abs(L_temp.spot_angle - spot_angle) > 0.001) continue;
        
        // 聚光混合判断
        if(spot_blend > 0.0 && abs(L_temp.spot_blend - spot_blend) > 0.001) continue;
        
        // 半径判断
        
        if(radius > 0.0 && distance(position, L_temp.position) > radius) continue;
        // 找到第一个满足条件的灯光
        L = L_temp;
        L.direction=-L.direction;
        found = true;
        break;
    }
    
}
#endif

/* META
@a: default=(0.0,0.0,0.0);
@b: default=(0.0,0.0,0.0);
@tolerance: default=0.0;
*/
bool colorsEqual(vec3 a, vec3 b, float tolerance)
{
    // 颜色差值
    float diff = length(a - b);
    return diff <= tolerance;
}

/* META
@dir: default=view_direction();
@scale: default=1.0;
*/
void toSphericalCoordinates(in vec3 dir,in float scale,in bool Clamp, out vec2 uv)
{
    // 确保dir是单位向量
    vec3 n = normalize(dir);
    uv = normalize(vec2(dir.x,dir.y));
    float theta = atan(n.z,sqrt(n.x * n.x + n.y * n.y));
    theta = float_remap(theta,0.0,PI/2.0,1.0,0.0);
    if(Clamp)
    {
        theta = clamp(theta,0.0,1.0);
    }
    uv *= theta*0.5*scale;
    uv+=0.5;
}

vec3 rotateToAlignZ(vec3 pos, vec3 targetDir) {
    vec3 z = normalize(targetDir);
    vec3 up = vec3(0.0, 0.0, 1.0); // Fallback up vector
    
    // Compute new x-axis (cross product of up and z)
    vec3 x = normalize(cross(up, z));
    
    // Compute new y-axis (cross product of z and x)
    vec3 y = cross(z, x);
    
    // Construct rotation matrix
    mat3 rotation = mat3(x, y, z);
    
    // Apply rotation to position
    return rotation * pos;
}

vec2 applyParallax(vec3 viewDir, vec2 uv, float heightScale,float height) {
    // Normalize view direction
    vec3 viewDirNorm = normalize(viewDir);
    
    // Initial height and UV
    vec2 p = viewDirNorm.xy * (height * heightScale) / max(viewDirNorm.z, 0.001); // Avoid division by zero
    vec2 newUV = uv + p;
    
    // Clamp UV to prevent sampling outside texture
    newUV = clamp(newUV, 0.0, 1.0);
    return newUV;
}

/* META
@uv: default=UV[0];
@tangent: default=get_tangent(0);
@normal: default=NORMAL;
@view_dir: default=view_direction();
@height_scale: default=1.0;
@layers: default=10;min=1;
@invert: default=false;
@occlusion: default=false;
@uv_out: default=(0.0,0.0);
*/
void Steep_parallax(in sampler2D height_map,in vec2 uv,in vec3 tangent,in vec3 normal,in vec3 view_dir,in float height_scale,in int layers,in bool invert,in bool occlusion,out vec2 uv_out)
{
    //陡峭视差
    vec3 v=normalize(view_dir);
    mat3 TBN=mat3(tangent,cross(normal,tangent),normal);
    vec3 VTS=normalize(v*TBN);
    vec2 offset=(VTS.xy/VTS.z*height_scale)/(layers*10);
    vec2 delta=offset;
    vec2 newUV=uv;
    float height0=0,height1=invert?texture(height_map,newUV).r:1.-texture(height_map,newUV).r;
    while(height0<height1)
    {
        newUV-=delta;
        height0+=1./layers;
        height1=invert?texture(height_map,newUV).r:1.-texture(height_map,newUV).r;
    }
    uv_out=newUV;
    if(occlusion)
    {
        // 视差遮蔽映射
        vec2 oldUV=newUV+delta;
        float height1_re=invert?texture(height_map,oldUV).r:1.-texture(height_map,oldUV).r;
        float height0_re=height0-1./layers;
        
        float heightAfter=height1-height0;
        float heightBefore=height1_re-height0_re;
        
        float weight=heightAfter/(heightAfter-heightBefore);
        newUV=oldUV*weight+newUV*(1-weight);
        uv_out=newUV;
    }
}

vec3 Log2Tonemap(vec3 color)
{
    // 避免log(0)
    vec3 logColor=log2(max(color,vec3(1e-4)));
    
    // 自定义压缩曲线
    vec3 compressed=exp2(logColor*.33)*1.4938-.7;
    
    // 根据亮度选择压缩或者原色
    vec3 mask=step(vec3(.3),color);// color > 0.3时选择压缩
    vec3 tonemapped=mix(color,compressed,mask);
    
    // 夹紧到0~1
    return clamp(tonemapped,0.,1.);
}

float hash1(float n)
{
    return fract(sin(n) * 43758.5453);
}

float noise4(in vec4 x)
{
    vec4 p = floor(x);  // 整数部分
    vec4 f = fract(x);  // 小数部分
    f = f * f * (3.0 - 2.0 * f);
    float n = p.x + p.y * 57.0 + p.z * 113.0 + p.w * 347.0;
    float res = mix(
        mix(
            mix(
                mix(hash1(n + 0.0), hash1(n + 1.0), f.x),
                mix(hash1(n + 57.0), hash1(n + 58.0), f.x), f.y
            ),
            mix(
                mix(hash1(n + 113.0), hash1(n + 114.0), f.x),
                mix(hash1(n + 170.0), hash1(n + 171.0), f.x), f.y
            ), f.z
        ),
        mix(
            mix(
                mix(hash1(n + 347.0), hash1(n + 348.0), f.x),
                mix(hash1(n + 404.0), hash1(n + 405.0), f.x), f.y
            ),
            mix(
                mix(hash1(n + 460.0), hash1(n + 461.0), f.x),
                mix(hash1(n + 517.0), hash1(n + 518.0), f.x), f.y
            ), f.z
        ), f.w
    );
    return res;
}

float fbm4(vec3 pos, float time, int octaves, float gain, float lacunarity)
{
    float f = 0.0;         // 最终输出值
    float amplitude = 0.5; // 初始振幅
    float frequency = 1.0; // 初始频率
    vec4 p = vec4(pos, time);
    // 旋转矩阵（扩展到四维）
    mat4 m = mat4(
        0.0,  0.80, 0.60, 0.0,
        -0.80, 0.36, -0.48, 0.0,
        -0.60, -0.48, 0.64, 0.0,
        0.0,  0.0,  0.0,  1.0
    );
    for (int i = 0; i < octaves; i++)
    {
        f += amplitude * noise4(p * frequency);
        p = m * p * lacunarity; // 频率倍增
        amplitude *= gain;       // 振幅衰减
    }
    return f;
}

float sdSphere(vec3 p,float radius){
    return-(length(p)-radius);
}
// SDF for a finite cylinder along an axis
float sdCylinder(vec3 p,float radius){
    return-(length(p*vec3(1.,0.,1.))-radius);
}
float sdBox(vec3 p, vec3 b)
{
    vec3 q = abs(p) - b;
    return -(length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0));
}

float smin(float a, float b, float k)
{
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// ----------------- 体积密度 -----------------
float computeDensity(vec3 p,float iTime,vec3 warp){
    float density=sdSphere(p*vec3(1.,3.,1.),1.5+(fbm4(p*warp.z+vec3(iTime*.2,0.,iTime*.2),iTime/5.,6,.5,2.02)-.5)*warp.x);
    // float density=sdSphere(p*vec3(1.,1.5,1.),1.);
    density=max(sdCylinder(p,.05)*10.,density);
    density=max(sdCylinder(p.yxz,.05)*10.,density);
    density=max(sdCylinder(p.yzx,.05)*10.,density);
    density = smin(density, clamp(sdBox(p, vec3(1.5)),0.,1.),warp.y);
    return clamp(density*1.,0.,1.);
}
// ----------------- AABB 交点 -----------------
vec2 intersectBox(vec3 ro,vec3 rd,vec3 boxMin,vec3 boxMax){
    vec3 tMin=(boxMin-ro)/rd;
    vec3 tMax=(boxMax-ro)/rd;
    vec3 t1=min(tMin,tMax);
    vec3 t2=max(tMin,tMax);
    float tNear=max(max(t1.x,t1.y),t1.z);
    float tFar=min(min(t2.x,t2.y),t2.z);
    return vec2(tNear,tFar);
}
// 射线与坐标轴的交点检测
float intersectAxis(vec3 rayOrigin,vec3 rayDir,vec3 axisDir,float axisLength){
    // 计算射线与坐标轴的交点
    float t=dot(axisDir,(axisDir*axisLength-rayOrigin))/dot(rayDir,axisDir);
    return t;
}
// ----------------- 体积 Ray March -----------------
/* META
@sun_pos: default=vec3(0.,2.,2.);
@sun_color: default=vec3(1.,.6706,.6706);
@shadow_color: default=vec3(.3843,.4941,.9333);
@color_cloud: default=vec3(.4588,.6471,.9255);
@transmittance: default=7.0;
@shadow_multiplier: default=15.0;
@cameraPos: default=camera_position();
@worldPos: default=transform_point(inverse(MODEL), POSITION);
@box_size: default=1.;
@max_step: default=128;
@shadow_step: default=16;
@Time: default=0.;
@warp: default=vec3(8.5,1.,1.5);
*/
void cloud_ray_marching(out vec4 cloud_color,out float cloudDepth,vec3 sun_pos,vec3 sun_color,vec3 shadow_color,vec3 color_cloud,float transmittance,float shadow_multiplier,vec3 cameraPos,vec3 worldPos,float box_size,int max_step,int shadow_step,float Time,vec3 warp){
    worldPos=worldPos.xzy;
    worldPos.y=-worldPos.y;
    cameraPos=cameraPos.xzy;
    cameraPos.y=-cameraPos.y;
    sun_pos=sun_pos.xzy;
    sun_pos.y=-sun_pos.y;
    float iTime=Time;
    vec4 color=vec4(0.,0.,0.,1.);
    vec3 ro=cameraPos;
    vec3 rd=normalize(worldPos-cameraPos);
    
    vec2 hit=intersectBox(ro,rd,vec3(-box_size),vec3(box_size));
    if(hit.x>hit.y||hit.y<0.){
        cloud_color=color;
        return;
    }
    
    float tStart=max(hit.x,0.);
    float tEnd=hit.y;
    float stepSize=(tEnd-tStart)/float(max_step);
    
    // vec3 sun_pos=vec3(0.,2.,2.);
    // vec3 sun_color=vec3(1.,.6706,.6706)*5.;
    // vec3 shadow_color=vec3(.3843,.4941,.9333)*3.;
    vec4 sum=vec4(0.,0.,0.,1.);//a通道是透过率
    vec3 pos=ro+rd*tStart;
    for(int i=0;i<max_step;i++){
        if(sum.a<.1)//如果透过率才小就直接返回
        {
            break;
        }
        float jitter=(hash1(float(i))-.5);
        pos+=rd*(stepSize+jitter*.1*stepSize);
        // pos+=rd*stepSize+vec3(hash1(float(i)),hash1(float(i+1)),hash1(float(i+2)))*0.01;//当前的位置
        float density=computeDensity(pos,iTime,warp);//获取当前位置的密度
        if(density>.01)
        {
            vec3 lpo=pos;
            float shadow=0.;
            for(int j=0;j<shadow_step;j++)
            {
                vec3 lightdir=normalize(sun_pos-lpo);
                lpo+=lightdir*stepSize;
                shadow+=computeDensity(lpo,iTime,warp)*2.;
            }
            density*=.1;
            float s=exp((-shadow/float(shadow_step))*shadow_multiplier);//当前点的光线强度 光源遮挡越多 shadow越大 s越接近0 越黑
            s=pow(s,1.);
            sum.rgb+=vec3(s*density)*sun_color*sum.a;//散射强度 密度*光强*光源颜色*透过率
            sum.a*=1.-density*transmittance;//更新透过率
            sum.rgb+=(density*shadow_color*sum.a);
        }
        density=clamp(density,0.,1.);
        sum+=density*stepSize*2.*vec4(color_cloud,1.);
        // 第一次达到深度阈值，记录深度
        if(sum.a>.9){
            cloudDepth=length(pos-cameraPos);// 摄像机到云表面的距离
        }
    }
    sum=clamp(sum,0.,1.);
    color=sum;
    color.a=1-color.a;
    cloud_color=color;
}
