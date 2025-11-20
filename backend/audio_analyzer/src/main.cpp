#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct AnalyzerConfig
{
    double silenceThresholdDb = -42.0;
    double lowEnergyThresholdDb = -34.0;
    double minSilenceDuration = 0.30; // seconds
    double minSpeechDuration = 0.15;  // seconds
    double windowMs = 25.0;
    double hopMs = 10.0;
    double preRoll = 0.12;
    double postRoll = 0.15;
};

struct AudioBuffer
{
    uint32_t sampleRate = 0;
    uint16_t channels = 0;
    std::vector<float> samples; // mono

    [[nodiscard]] double duration() const
    {
        if (sampleRate == 0)
        {
            return 0.0;
        }
        return static_cast<double>(samples.size()) / static_cast<double>(sampleRate);
    }
};

struct FrameInfo
{
    double startTime = 0.0;
    double endTime = 0.0;
    double rms = 0.0;
};

struct Segment
{
    double start = 0.0;
    double end = 0.0;
    double peak = 0.0;
    double avg = 0.0;
};

template <typename T>
T readLittleEndian(std::ifstream& stream)
{
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!stream)
    {
        throw std::runtime_error("Unexpected end of file while reading WAV file.");
    }
    return value;
}

AudioBuffer loadWav(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("Unable to open WAV file: " + path);
    }

    const uint32_t riff = readLittleEndian<uint32_t>(input);
    const uint32_t totalSize = readLittleEndian<uint32_t>(input);
    (void)totalSize;
    const uint32_t wave = readLittleEndian<uint32_t>(input);
    if (riff != 0x46464952 || wave != 0x45564157) // "RIFF" "WAVE"
    {
        throw std::runtime_error("File is not a valid PCM WAV.");
    }

    uint16_t audioFormat = 0;
    uint16_t channels = 0;
    uint32_t sampleRate = 0;
    uint16_t bitsPerSample = 0;
    std::vector<char> rawData;

    while (input && static_cast<std::size_t>(input.tellg()) < (8 + totalSize))
    {
        uint32_t chunkId = 0;
        uint32_t chunkSize = 0;
        input.read(reinterpret_cast<char*>(&chunkId), sizeof(uint32_t));
        input.read(reinterpret_cast<char*>(&chunkSize), sizeof(uint32_t));
        if (!input)
        {
            break;
        }

        const std::streampos nextChunkPos = input.tellg() + static_cast<std::streamoff>(chunkSize);
        if (chunkId == 0x20746D66) // "fmt "
        {
            audioFormat = readLittleEndian<uint16_t>(input);
            channels = readLittleEndian<uint16_t>(input);
            sampleRate = readLittleEndian<uint32_t>(input);
            const uint32_t byteRate = readLittleEndian<uint32_t>(input);
            (void)byteRate;
            const uint16_t blockAlign = readLittleEndian<uint16_t>(input);
            (void)blockAlign;
            bitsPerSample = readLittleEndian<uint16_t>(input);
        }
        else if (chunkId == 0x61746164) // "data"
        {
            rawData.resize(chunkSize);
            input.read(rawData.data(), chunkSize);
        }
        else
        {
            input.seekg(nextChunkPos);
        }

        input.seekg(nextChunkPos);
    }

    if (audioFormat != 1 && audioFormat != 3) // PCM or float32
    {
        throw std::runtime_error("Only PCM (int) or IEEE float WAV files are supported.");
    }
    if (channels == 0 || sampleRate == 0 || bitsPerSample == 0 || rawData.empty())
    {
        throw std::runtime_error("WAV header is incomplete.");
    }

    AudioBuffer buffer;
    buffer.sampleRate = sampleRate;
    buffer.channels = channels;

    const std::size_t bytesPerSample = bitsPerSample / 8;
    const std::size_t totalSamples = rawData.size() / bytesPerSample;
    const std::size_t frames = totalSamples / channels;
    buffer.samples.resize(frames);

    for (std::size_t frame = 0; frame < frames; ++frame)
    {
        double acc = 0.0;
        for (uint16_t ch = 0; ch < channels; ++ch)
        {
            const std::size_t idx = (frame * channels + ch) * bytesPerSample;
            double sampleValue = 0.0;
            if (audioFormat == 1)
            {
                if (bitsPerSample == 16)
                {
                    int16_t sample = 0;
                    std::memcpy(&sample, rawData.data() + idx, sizeof(sample));
                    sampleValue = static_cast<double>(sample) / static_cast<double>(std::numeric_limits<int16_t>::max());
                }
                else if (bitsPerSample == 24)
                {
                    const auto b0 = static_cast<unsigned char>(rawData[idx]);
                    const auto b1 = static_cast<unsigned char>(rawData[idx + 1]);
                    const auto b2 = static_cast<unsigned char>(rawData[idx + 2]);
                    int32_t sample = b0 | (b1 << 8) | (b2 << 16);
                    if (sample & 0x800000)
                    {
                        sample |= ~0xFFFFFF;
                    }
                    sampleValue = static_cast<double>(sample) / 8388607.0;
                }
                else if (bitsPerSample == 32)
                {
                    int32_t sample = 0;
                    std::memcpy(&sample, rawData.data() + idx, sizeof(sample));
                    sampleValue = static_cast<double>(sample) / static_cast<double>(std::numeric_limits<int32_t>::max());
                }
            }
            else if (audioFormat == 3 && bitsPerSample == 32)
            {
                float sample = 0.0F;
                std::memcpy(&sample, rawData.data() + idx, sizeof(sample));
                sampleValue = static_cast<double>(sample);
            }
            acc += sampleValue;
        }
        buffer.samples[frame] = static_cast<float>(acc / static_cast<double>(channels));
    }

    buffer.channels = 1;
    return buffer;
}

std::vector<FrameInfo> buildFrames(const AudioBuffer& buffer, const AnalyzerConfig& cfg)
{
    const std::size_t windowSamples = std::max<std::size_t>(
        1, static_cast<std::size_t>((cfg.windowMs / 1000.0) * buffer.sampleRate));
    const std::size_t hopSamples = std::max<std::size_t>(
        1, static_cast<std::size_t>((cfg.hopMs / 1000.0) * buffer.sampleRate));

    std::vector<FrameInfo> frames;
    frames.reserve(buffer.samples.size() / hopSamples + 1);

    for (std::size_t start = 0; start < buffer.samples.size(); start += hopSamples)
    {
        const std::size_t end = std::min(start + windowSamples, buffer.samples.size());
        if (end <= start)
        {
            break;
        }
        double energy = 0.0;
        double peak = 0.0;
        for (std::size_t i = start; i < end; ++i)
        {
            const double sample = buffer.samples[i];
            energy += sample * sample;
            peak = std::max(peak, std::abs(sample));
        }
        const double windowSize = static_cast<double>(end - start);
        double rms = std::sqrt(energy / windowSize);
        if (peak < 1e-9)
        {
            rms = 0.0;
        }

        FrameInfo info;
        info.startTime = static_cast<double>(start) / static_cast<double>(buffer.sampleRate);
        info.endTime = static_cast<double>(end) / static_cast<double>(buffer.sampleRate);
        info.rms = rms;
        frames.push_back(info);
    }
    return frames;
}

std::vector<Segment> framesToSegments(const std::vector<FrameInfo>& frames,
                                      const AnalyzerConfig& cfg,
                                      const bool targetSpeech,
                                      const double threshold)
{
    std::vector<Segment> segments;
    bool state = false;
    double segStart = 0.0;
    double peak = 0.0;
    double acc = 0.0;
    std::size_t accCount = 0;

    const double minDuration = targetSpeech ? cfg.minSpeechDuration : cfg.minSilenceDuration;

    for (std::size_t i = 0; i < frames.size(); ++i)
    {
        const auto& frame = frames[i];
        const double value = frame.rms;
        const bool isSpeechFrame = value >= threshold;
        const bool matches = targetSpeech ? isSpeechFrame : (value < threshold);

        if (!state && matches)
        {
            state = true;
            segStart = frame.startTime;
            peak = value;
            acc = value;
            accCount = 1;
        }
        else if (state && matches)
        {
            peak = std::max(peak, value);
            acc += value;
            ++accCount;
        }
        else if (state && !matches)
        {
            const double segEnd = frames[i - 1].endTime;
            const double duration = segEnd - segStart;
            if (duration >= minDuration)
            {
                Segment segment;
                segment.start = segStart;
                segment.end = segEnd;
                segment.peak = peak;
                segment.avg = accCount > 0 ? acc / static_cast<double>(accCount) : 0.0;
                segments.push_back(segment);
            }
            state = false;
        }
    }

    if (state)
    {
        const double segEnd = frames.back().endTime;
        const double duration = segEnd - segStart;
        if (duration >= minDuration)
        {
            Segment segment;
            segment.start = segStart;
            segment.end = segEnd;
            segment.peak = peak;
            segment.avg = accCount > 0 ? acc / static_cast<double>(accCount) : 0.0;
            segments.push_back(segment);
        }
    }

    // Merge adjacent segments that are very close to avoid jitter
    if (!segments.empty())
    {
        std::vector<Segment> merged;
        merged.reserve(segments.size());
        merged.push_back(segments.front());
        for (std::size_t i = 1; i < segments.size(); ++i)
        {
            auto& last = merged.back();
            const auto& current = segments[i];
            if (current.start - last.end < 0.05)
            {
                last.end = current.end;
                last.peak = std::max(last.peak, current.peak);
                last.avg = (last.avg + current.avg) * 0.5;
            }
            else
            {
                merged.push_back(current);
            }
        }
        segments.swap(merged);
    }

    return segments;
}

std::vector<Segment> withPadding(const std::vector<Segment>& segments,
                                 const double duration,
                                 const double preRoll,
                                 const double postRoll)
{
    std::vector<Segment> padded;
    padded.reserve(segments.size());
    for (const auto& seg : segments)
    {
        Segment adjusted = seg;
        adjusted.start = std::max(0.0, adjusted.start - preRoll);
        adjusted.end = std::min(duration, adjusted.end + postRoll);
        padded.push_back(adjusted);
    }
    return padded;
}

std::string escape(const std::string& input)
{
    std::ostringstream oss;
    for (char ch : input)
    {
        switch (ch)
        {
            case '"':
                oss << "\\\"";
                break;
            case '\\':
                oss << "\\\\";
                break;
            case '\b':
                oss << "\\b";
                break;
            case '\f':
                oss << "\\f";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20)
                {
                    oss << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(ch))
                        << std::dec << std::setw(0);
                }
                else
                {
                    oss << ch;
                }
        }
    }
    return oss.str();
}

void printSegments(std::ostringstream& oss, const std::vector<Segment>& segments)
{
    oss << "[";
    for (std::size_t i = 0; i < segments.size(); ++i)
    {
        const auto& seg = segments[i];
        oss << "{"
            << "\"start\":" << std::fixed << std::setprecision(6) << seg.start << ","
            << "\"end\":" << std::fixed << std::setprecision(6) << seg.end << ","
            << "\"peak\":" << std::fixed << std::setprecision(6) << seg.peak << ","
            << "\"avg\":" << std::fixed << std::setprecision(6) << seg.avg
            << "}";
        if (i + 1 < segments.size())
        {
            oss << ",";
        }
    }
    oss << "]";
}

void emitJson(const AudioBuffer& buffer,
              const std::vector<FrameInfo>& frames,
              const std::vector<Segment>& speechSegments,
              const std::vector<Segment>& silenceSegments,
              const std::vector<Segment>& lowSegments,
              const AnalyzerConfig& cfg)
{
    std::ostringstream oss;
    oss << "{";
    oss << "\"sample_rate\":" << buffer.sampleRate << ",";
    oss << "\"duration\":" << std::fixed << std::setprecision(6) << buffer.duration() << ",";
    oss << "\"config\":{"
        << "\"threshold_db\":" << cfg.silenceThresholdDb << ","
        << "\"low_energy_threshold_db\":" << cfg.lowEnergyThresholdDb << ","
        << "\"window_ms\":" << cfg.windowMs << ","
        << "\"hop_ms\":" << cfg.hopMs
        << "},";

    oss << "\"envelope\":[";
    for (std::size_t i = 0; i < frames.size(); ++i)
    {
        const auto& frame = frames[i];
        oss << "{"
            << "\"time\":" << std::fixed << std::setprecision(6) << frame.startTime << ","
            << "\"rms\":" << std::fixed << std::setprecision(6) << frame.rms
            << "}";
        if (i + 1 < frames.size())
        {
            oss << ",";
        }
    }
    oss << "],";

    oss << "\"segments\":{";
    oss << "\"speech\":"; printSegments(oss, speechSegments); oss << ",";
    oss << "\"silence\":"; printSegments(oss, silenceSegments); oss << ",";
    oss << "\"low_energy\":"; printSegments(oss, lowSegments);
    oss << "},";

    // Provide mask suggestions for silence + low energy segments
    oss << "\"mask_suggestions\":[";
    bool first = true;
    const auto emitSuggestion = [&](const std::vector<Segment>& list, const std::string& label) {
        for (const auto& seg : list)
        {
            if (!first)
            {
                oss << ",";
            }
            oss << "{"
                << "\"type\":\"" << escape(label) << "\","
                << "\"start\":" << std::fixed << std::setprecision(6) << seg.start << ","
                << "\"end\":" << std::fixed << std::setprecision(6) << seg.end
                << "}";
            first = false;
        }
    };
    emitSuggestion(silenceSegments, "silence");
    emitSuggestion(lowSegments, "low_energy");
    oss << "]";
    oss << "}";

    std::cout << oss.str() << std::endl;
}

AnalyzerConfig parseArgs(int argc, char** argv, std::string& inputPath)
{
    AnalyzerConfig config;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        const auto readValue = [&](double& target) {
            if (i + 1 >= argc)
            {
                throw std::runtime_error("Missing value for argument: " + arg);
            }
            target = std::stod(argv[++i]);
        };

        if (arg == "--input" || arg == "-i")
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error("Missing value for --input");
            }
            inputPath = argv[++i];
        }
        else if (arg == "--threshold-db")
        {
            readValue(config.silenceThresholdDb);
        }
        else if (arg == "--low-db")
        {
            readValue(config.lowEnergyThresholdDb);
        }
        else if (arg == "--min-silence")
        {
            readValue(config.minSilenceDuration);
        }
        else if (arg == "--min-speech")
        {
            readValue(config.minSpeechDuration);
        }
        else if (arg == "--window-ms")
        {
            readValue(config.windowMs);
        }
        else if (arg == "--hop-ms")
        {
            readValue(config.hopMs);
        }
        else if (arg == "--pre-roll")
        {
            readValue(config.preRoll);
        }
        else if (arg == "--post-roll")
        {
            readValue(config.postRoll);
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (inputPath.empty())
    {
        throw std::runtime_error("Missing --input <wav file> argument.");
    }
    return config;
}

} // namespace

int main(int argc, char** argv)
{
    std::string inputPath;
    AnalyzerConfig config;
    try
    {
        config = parseArgs(argc, argv, inputPath);
        const AudioBuffer buffer = loadWav(inputPath);
        const auto frames = buildFrames(buffer, config);
        if (frames.empty())
        {
            throw std::runtime_error("Audio buffer too short for analysis.");
        }
        const double silenceThreshold = std::pow(10.0, config.silenceThresholdDb / 20.0);
        const double lowEnergyThreshold = std::pow(10.0, config.lowEnergyThresholdDb / 20.0);
        const auto speechSegments = withPadding(
            framesToSegments(frames, config, true, silenceThreshold),
            buffer.duration(),
            config.preRoll,
            config.postRoll);
        const auto silenceSegments = framesToSegments(frames, config, false, silenceThreshold);

        std::vector<Segment> lowSegments;
        lowSegments.reserve(frames.size());
        bool inLow = false;
        double lowStart = 0.0;
        double peak = 0.0;
        double acc = 0.0;
        std::size_t count = 0;
        for (const auto& frame : frames)
        {
            const bool isLow = frame.rms >= silenceThreshold && frame.rms < lowEnergyThreshold;
            if (isLow && !inLow)
            {
                inLow = true;
                lowStart = frame.startTime;
                peak = frame.rms;
                acc = frame.rms;
                count = 1;
            }
            else if (isLow && inLow)
            {
                peak = std::max(peak, frame.rms);
                acc += frame.rms;
                ++count;
            }
            else if (!isLow && inLow)
            {
                const double lowEnd = frame.startTime;
                const double duration = lowEnd - lowStart;
                if (duration >= config.minSpeechDuration * 0.5)
                {
                    Segment seg;
                    seg.start = lowStart;
                    seg.end = lowEnd;
                    seg.peak = peak;
                    seg.avg = count > 0 ? acc / static_cast<double>(count) : 0.0;
                    lowSegments.push_back(seg);
                }
                inLow = false;
            }
        }
        if (inLow)
        {
            Segment seg;
            seg.start = lowStart;
            seg.end = frames.back().endTime;
            seg.peak = peak;
            seg.avg = count > 0 ? acc / static_cast<double>(count) : 0.0;
            lowSegments.push_back(seg);
        }

        emitJson(buffer, frames, speechSegments, silenceSegments, lowSegments, config);
    }
    catch (const std::exception& ex)
    {
        std::cerr << "{"
                  << "\"error\":\"" << escape(ex.what()) << "\""
                  << "}" << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
