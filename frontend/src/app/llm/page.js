"use client"

import { useState } from 'react';
import axios from 'axios';
import { Box, Input, Button, Text } from '@chakra-ui/react';

const LLM = () => {
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");

  const handleGenerate = async () => {
    try {
      const res = await axios.post(
        'http://127.0.0.1:5000/generate', // Include the '/generate' endpoint
        { prompt }, // Send the prompt as the request body
        {
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );
      setResponse(res.data.response);
    } catch (error) {
      console.error('Error generating text:', error);
    }
  };
  

  return (
    <Box p={4}>
      <Input
        placeholder="Enter your prompt"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        mb={4}
      />
      <Button onClick={handleGenerate} colorScheme="teal">Generate</Button>
      {response && <Text mt={4} p={2} border="1px solid" borderRadius="md">{response}</Text>}
    </Box>
  );
};

export default LLM;
