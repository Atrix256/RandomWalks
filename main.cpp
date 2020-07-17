// ------------------ SETTINGS ------------------

// Counts to show of the single example random walk
static const size_t c_randomWalkVisSteps[] =
{
    10,
    100,
    1000,
    //10000
};
static const size_t c_randomWalkVisImageSize = 256;

#define DETERMINISTIC() true

// ----------------------------------------------

#define _CRT_SECURE_NO_WARNINGS

#include <random>
#include <array>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

// ------------- Constants and Typedefs -------------------

#define COUNT_OF(array) (sizeof(array) / sizeof(array[0]))

typedef std::array<float, 2> Vec2;
typedef std::array<int, 2> Vec2i;

static const size_t c_randomWalkVisStepsCount = COUNT_OF(c_randomWalkVisSteps);
static const size_t c_randomWalkVisStepsMax = c_randomWalkVisSteps[c_randomWalkVisStepsCount - 1];

static const float c_pi = 3.14159265359f;
static const float c_piFract = 0.14159265359f;
static const float c_goldenRatio = 1.61803398875f;
static const float c_goldenRatioConjugate = 0.61803398875f;
static const float c_sqrt2 = 1.41421356237f;
static const float c_sqrt2Fract = 0.41421356237f;

// --------------- Utils ---------------------

template <typename T>
T Clamp(T value, T min, T max)
{
    if (value <= min)
        return min;
    else if (value >= max)
        return max;
    else
        return value;
}

template <typename T, size_t N>
std::array<T, N> Clamp(const std::array<T, N>& value, T min, T max)
{
    std::array<T, N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = Clamp(value[i], min, max);
    return ret;
}

template <typename T>
T Max(T a, T b)
{
    return a >= b ? a : b;
}

template <typename T>
T Min(T a, T b)
{
    return a < b ? a : b;
}

float Fract(float f)
{
    return f - floor(f);
}

template <size_t N>
float Dot(const std::array<float, N>& A, const std::array<float, N>& B)
{
    float ret = 0.0f;
    for (size_t i = 0; i < N; ++i)
        ret += A[i] * B[i];
    return ret;
}

template <size_t N>
float Length(const std::array<float, N>& V)
{
    float ret = 0.0f;
    for (float f : V)
        ret += f * f;
    return sqrtf(ret);
}

float Lerp(float A, float B, float t)
{
    return A * (1.0f - t) + B * t;
}

float SmoothStep(float min, float max, float x)
{
    if (x <= min)
        return 0.0f;
    else if (x >= max)
        return 1.0f;

    float percent = (x - min) / (max - min);
    return 3.0f * percent * percent - 2.0f * percent * percent * percent;
}

template <typename T, size_t N>
std::array<T, N> operator - (const std::array<T, N>& A, const std::array<T, N>& B)
{
    std::array<T, N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A[i] - B[i];
    return ret;
}

template <typename T, size_t N>
std::array<T, N> operator * (const std::array<T, N>& A, T B)
{
    std::array<T, N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A[i] * B;
    return ret;
}

std::mt19937 GetRNG()
{
#if DETERMINISTIC()
    static int seed = 1336;
    seed++;
    std::mt19937 rng(seed);
#else
    std::random_device rd;
    std::mt19937 rng(rd());
#endif
    return rng;
}

// ---------------- Drawing Utils -----------------

struct RGB
{
    unsigned char R, G, B;
};

RGB Lerp(const RGB& A, const RGB& B, float t)
{
    return RGB
    {
        (unsigned char)Lerp((float)A.R, (float)B.R, t),
        (unsigned char)Lerp((float)A.G, (float)B.G, t),
        (unsigned char)Lerp((float)A.B, (float)B.B, t),
    };
}

// https://www.iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
float sdSegment(const Vec2& p, const Vec2& a, const Vec2& b)
{
    Vec2 pa = p - a, ba = b - a;
    float h = Clamp(Dot(pa, ba) / Dot(ba, ba), 0.0f, 1.0f);
    return Length(pa - ba * h);
}

void DrawLine(std::vector<RGB>& pixels, const Vec2i& dims, const Vec2i& start, const Vec2i& end, const RGB& color)
{
    // draws a line via SDF. Not the most efficient, but it works!

    // get a padded min / max
    Vec2i min =
    {
        Clamp(Min(start[0], end[0])-2, 0, dims[0] - 1),
        Clamp(Min(start[1], end[1])-2, 0, dims[0] - 1),
    };

    Vec2i max =
    {
        Clamp(Max(start[0], end[0])+2, 0, dims[0] - 1),
        Clamp(Max(start[1], end[1])+2, 0, dims[0] - 1),
    };

    for (size_t y = min[1]; y <= max[1]; ++y)
    {
        RGB* pixel = &pixels[y * dims[0] + min[0]];
        for (size_t x = min[0]; x <= max[0]; ++x)
        {
            float dist = sdSegment(Vec2{ float(x), float(y) }, Vec2{ float(start[0]), float(start[1]) }, Vec2{ float(end[0]), float(end[1]) });
            float AAFade = 1.0f - SmoothStep(0.0f, 2.0f, dist);
            if (AAFade > 0.0f)
                *pixel = Lerp(*pixel, color, AAFade);

            pixel++;
        }
    }
}

// ----------------------------------------------

static void BlueNoise_BestCandidate(std::vector<float>& values, size_t numValues, std::mt19937& rng)
{
    // if they want less samples than there are, just truncate the sequence
    if (numValues <= values.size())
    {
        values.resize(numValues);
        return;
    }

    static std::uniform_real_distribution<float> dist(0, 1);

    // handle the special case of not having any values yet, so we don't check for it in the loops.
    if (values.size() == 0)
        values.push_back(dist(rng));

    // make a sorted list of existing samples
    std::vector<float> sortedValues;
    sortedValues = values;
    sortedValues.reserve(numValues);
    values.reserve(numValues);
    std::sort(sortedValues.begin(), sortedValues.end());

    // use whatever samples currently exist, and just add to them, since this is a progressive sequence
    for (size_t i = values.size(); i < numValues; ++i)
    {
        size_t numCandidates = values.size();
        float bestDistance = 0.0f;
        float bestCandidateValue = 0;
        size_t bestCandidateInsertLocation = 0;
        for (size_t candidate = 0; candidate < numCandidates; ++candidate)
        {
            float candidateValue = dist(rng);

            // binary search the sorted value list to find the values it's closest to.
            auto lowerBound = std::lower_bound(sortedValues.begin(), sortedValues.end(), candidateValue);
            size_t insertLocation = lowerBound - sortedValues.begin();

            // calculate the closest distance (torroidally) from this point to an existing sample by looking left and right.
            float distanceLeft = (insertLocation > 0)
                ? candidateValue - sortedValues[insertLocation - 1]
                : 1.0f + candidateValue - *sortedValues.rbegin();

            float distanceRight = (insertLocation < sortedValues.size())
                ? sortedValues[insertLocation] - candidateValue
                : distanceRight = 1.0f + sortedValues[0] - candidateValue;

            // whichever is closer left vs right is the closer point distance
            float minDist = std::min(distanceLeft, distanceRight);

            // keep the best candidate seen (maximize the closest distance)
            if (minDist > bestDistance)
            {
                bestDistance = minDist;
                bestCandidateValue = candidateValue;
                bestCandidateInsertLocation = insertLocation;
            }
        }

        // take the best candidate and also insert it into the sorted values
        sortedValues.insert(sortedValues.begin() + bestCandidateInsertLocation, bestCandidateValue);
        values.push_back(bestCandidateValue);
    }
}

template <bool VARIATION>
static void RedNoise_BestCandidate(std::vector<float>& values, size_t numValues, std::mt19937& rng)
{
    // if they want less samples than there are, just truncate the sequence
    if (numValues <= values.size())
    {
        values.resize(numValues);
        return;
    }

    static std::uniform_real_distribution<float> dist(0, 1);

    // handle the special case of not having any values yet, so we don't check for it in the loops.
    if (values.size() == 0)
        values.push_back(dist(rng));

    // use whatever samples currently exist, and just add to them, since this is a progressive sequence
    for (size_t i = values.size(); i < numValues; ++i)
    {
        size_t numCandidates = values.size();
        float bestDistance = FLT_MAX;
        float bestCandidateValue = 0;
        for (size_t candidate = 0; candidate < numCandidates; ++candidate)
        {
            float candidateValue = dist(rng);

            float maxDistance = 0.0f;
            for (size_t valueIndex = 0; valueIndex < values.size(); ++valueIndex)
            {
                float dist = abs(values[valueIndex] - candidateValue);
                if (dist > 0.5f)
                    dist = 1.0f - dist;

                if (VARIATION == false)
                {
                    // we want to minimize the distance to the farthest candidate
                    if (dist > maxDistance)
                        maxDistance = dist;
                }
                else
                {
                    // we want to minimize the distance to the closest candidate
                    if (dist < maxDistance)
                        maxDistance = dist;
                }
            }

            // keep the best candidate seen (minimize the closest distance)
            if (maxDistance < bestDistance)
            {
                bestDistance = maxDistance;
                bestCandidateValue = candidateValue;
            }
        }

        // take the best candidate
        values.push_back(bestCandidateValue);
    }
}

/*
TODO: how should red noise be generated?

* blue maximizes distance to nearest neighbor

* red noise could minimize distance to nearest neighbor
* it could also minimize distance to farthest neighbor

you thought you tried them both but you had a bug, you didn't!

*/

// -------------------- MAIN LOGIC --------------------------

void DrawRandomWalk(const char* label, const std::vector<Vec2>& steps)
{
    float maxAbsDimension = 0.0f;
    for (const Vec2& v : steps)
    {
        maxAbsDimension = Max(maxAbsDimension, abs(v[0]));
        maxAbsDimension = Max(maxAbsDimension, abs(v[1]));
    }

    // a helper to convert from mathematical coordinates to pixel coordinates
    auto CoordToPixel = [maxAbsDimension](const Vec2& coord) -> Vec2i
    {
        float percentX = Clamp((coord[0] + maxAbsDimension) / (maxAbsDimension * 2.0f), 0.0f, 1.0f);
        float percentY = Clamp((coord[1] + maxAbsDimension) / (maxAbsDimension * 2.0f), 0.0f, 1.0f);

        Vec2i ret =
        {
            Clamp<int>(int(percentX * float(c_randomWalkVisImageSize)), 0, c_randomWalkVisImageSize - 1),
            Clamp<int>(int(percentY * float(c_randomWalkVisImageSize)), 0, c_randomWalkVisImageSize - 1)
        };

        return ret;
    };

    // draw grid and origin
    Vec2i imageDims = Vec2i{ c_randomWalkVisImageSize, c_randomWalkVisImageSize };
    std::vector<RGB> pixels(c_randomWalkVisImageSize * c_randomWalkVisImageSize, RGB{ 255, 255, 255 });
    static const size_t c_halfImageSize = c_randomWalkVisImageSize / 2;
    int line = 1;
    bool drawMinorAxes = (CoordToPixel(Vec2{ 1.0f, 0.0f }) - CoordToPixel(Vec2{ 0.0f, 0.0f }))[0] > 5; // only draw minor axes if they are at least this many pixels apart
    while (float(line) < maxAbsDimension)
    {
        bool majorAxis = (line % 5) == 0;
        RGB color = majorAxis ? RGB{ 192, 192, 192} : RGB{ 228, 228, 228};

        if (majorAxis || drawMinorAxes)
        {
            Vec2i pixelOffset = CoordToPixel(Vec2{ float(line), float(line) });
            DrawLine(pixels, imageDims, Vec2i{ pixelOffset[0] , 0 }, Vec2i{ pixelOffset[0] , c_randomWalkVisImageSize }, color);
            DrawLine(pixels, imageDims, Vec2i{ 0, pixelOffset[0] }, Vec2i{ c_randomWalkVisImageSize, pixelOffset[0] }, color);

            if (line != 0)
            {
                pixelOffset = CoordToPixel(Vec2{ -float(line), -float(line) });
                DrawLine(pixels, imageDims, Vec2i{ pixelOffset[0] , 0 }, Vec2i{ pixelOffset[0] , c_randomWalkVisImageSize }, color);
                DrawLine(pixels, imageDims, Vec2i{ 0, pixelOffset[0] }, Vec2i{ c_randomWalkVisImageSize, pixelOffset[0] }, color);
            }
        }

        line++;
    }
    DrawLine(pixels, imageDims, Vec2i{ c_halfImageSize , 0 }, Vec2i{ c_halfImageSize , c_randomWalkVisImageSize }, RGB{ 192, 192, 255 });
    DrawLine(pixels, imageDims, Vec2i{ 0, c_halfImageSize }, Vec2i{ c_randomWalkVisImageSize, c_halfImageSize }, RGB{ 255, 192, 192 });

    // make the image
    Vec2i lastp = CoordToPixel(Vec2{ 0.0f, 0.0f });
    for (size_t stepIndex = 0; stepIndex < steps.size(); ++stepIndex)
    {
        // get the next point
        const Vec2& point = steps[stepIndex];
        Vec2i p = CoordToPixel(point);

        // draw a line
        RGB color;
        color.R = (unsigned char)Clamp(255.0f * (float(stepIndex) / float(steps.size()-1)), 0.0f, 255.0f);
        color.G = 0;
        color.B = 0;
        DrawLine(pixels, imageDims, lastp, p, color);

        // remember this point as the last point for the next time around
        lastp = p;
    }

    // save the image
    char fileName[256];
    sprintf_s(fileName, "out/%s_%zu.png", label, steps.size());
    stbi_write_png(fileName, c_randomWalkVisImageSize, c_randomWalkVisImageSize, 3, pixels.data(), c_randomWalkVisImageSize * 3);
}

template <typename GENERATOR>
void RandomWalkTest(const char* label, const GENERATOR& generator)
{
    printf("%s...\n", label);
    std::vector<Vec2> steps;
    Vec2 pos = { 0.0f, 0.0f };
    int lastPercent = -1;
    for (size_t stepIndex = 0; stepIndex < c_randomWalkVisStepsMax; ++stepIndex)
    {
        int percent = int(100.0f * float(stepIndex) / float(c_randomWalkVisStepsMax - 1));
        if (percent != lastPercent)
        {
            printf("\r%i%%", percent);
            lastPercent = percent;
        }
        // take a step 1 unit long in a random direction
        float angle = generator() * 2.0f * c_pi;
        pos[0] += cos(angle);
        pos[1] += sin(angle);
        steps.push_back(pos);

        // draw out the random walk when we should
        for (size_t reportIndex = 0; reportIndex < c_randomWalkVisStepsCount; ++reportIndex)
        {
            if (stepIndex == c_randomWalkVisSteps[reportIndex] - 1)
                DrawRandomWalk(label, steps);
        }
    }

    printf("\r100%%\n\n");
}

int main(int argc, char** argv)
{
    // white noise random walk test
    {
        std::mt19937 rng = GetRNG();
        RandomWalkTest("white",
            [&rng]()
            {
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                return dist(rng);
            }
        );
    }

    // golden ratio random walk test
    {
        float value = 0.0f;
        RandomWalkTest("GR", [&value]() { value = Fract(value + c_goldenRatioConjugate); return value; });
    }

    // alternating golden ratio random walk test
    // Makes an hour glass shape. not useful.
    if(false)
    {
        float value = 0.0f;
        bool flip = true;
        RandomWalkTest("GRFlip", [&value, &flip]()
            {
                flip = !flip;
                value = Fract(value + c_goldenRatioConjugate);
                return flip ? 1.0f - value : value;
            }
        );
    }

    // root 2 random walk test
    {
        float value = 0.0f;
        RandomWalkTest("Root2", [&value]() { value = Fract(value + c_sqrt2Fract); return value; });
    }

    // Pi random walk test
    {
        float value = 0.0f;
        RandomWalkTest("Pi", [&value]() { value = Fract(value + c_piFract); return value; });
    }

    // Blue Noise test
    {
        std::mt19937 rng = GetRNG();
        std::vector<float> blueNoise;
        size_t nextIndex = 0;
        RandomWalkTest("BlueNoise", [&blueNoise, &nextIndex, &rng]()
            {
                if (nextIndex == 0)
                    BlueNoise_BestCandidate(blueNoise, c_randomWalkVisStepsMax, rng);
                float ret = blueNoise[nextIndex];
                nextIndex++;
                return ret;
            }
        );
    }

    // Red Noise 1 test
    {
        std::mt19937 rng = GetRNG();
        std::vector<float> redNoise;
        size_t nextIndex = 0;
        RandomWalkTest("RedNoise1", [&redNoise, &nextIndex, &rng]()
            {
                if (nextIndex == 0)
                    RedNoise_BestCandidate<false>(redNoise, c_randomWalkVisStepsMax, rng);
                float ret = redNoise[nextIndex];
                nextIndex++;
                return ret;
            }
        );
    }

    // Red Noise 2 test
    {
        std::mt19937 rng = GetRNG();
        std::vector<float> redNoise;
        size_t nextIndex = 0;
        RandomWalkTest("RedNoise2", [&redNoise, &nextIndex, &rng]()
            {
                if (nextIndex == 0)
                    RedNoise_BestCandidate<true>(redNoise, c_randomWalkVisStepsMax, rng);
                float ret = redNoise[nextIndex];
                nextIndex++;
                return ret;
            }
        );
    }

    return 0;
}

/*

* get the binary search back in for red noise? definitely useful for finding the closest point, but can it find the farthest too? (closest to fract(value + 0.5)?)

? need to randomize LDS starting positions if doing multiple tests
* could draw the origin and a grid on the random walk tests to give some sense of location and distance
* could thread the multi tests for speedup. im betting best candiddate noise will be slow

? DFT the red noise to see?

Random walk single tests...
1) Show actual random walks - using different colors for different points in the line. rainbow, heatmap, ... ??

Random walk multi tests...
1) a histogram by distance from center (could show as rings around a circle i guess)
2) a histogram of angle?
3) a histogram breaking it up into both distance and angle?




Random walks... brain storming
1) white noise
2) golden ratio (should be bad)
3) alternating golden ratio
4) high discrepancy sequence?
5) pi?
6) grey code? bayer?

7) blue noise, red noise, HDS


High discrepancy sequences...
1) https://www.researchgate.net/publication/259156917_High-discrepancy_sequences
2) https://www.researchgate.net/publication/259156640_A_survey_of_high-discrepancy_sequences

Blog:
* how to make red noise

*/